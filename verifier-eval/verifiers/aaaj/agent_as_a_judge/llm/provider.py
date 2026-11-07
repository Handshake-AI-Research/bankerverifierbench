import warnings
import time
from functools import partial
import os
from dotenv import load_dotenv

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm

from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    BadRequestError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

os.environ["LITELLM_LOG"] = "DEBUG"

__all__ = ["LLM"]

message_separator = "\n\n----------\n\n"


def _retryable_provider_error(error):
    if isinstance(error, (RateLimitError, APIConnectionError, ServiceUnavailableError, Timeout)):
        return True
    # LiteLLM 1.83 can classify Google's HTTP 429 quota response as a
    # BadRequestError. Retry that transport condition, not genuine bad inputs.
    return isinstance(error, BadRequestError) and '"code": 429' in str(error) and 'RESOURCE_EXHAUSTED' in str(error)


class LLM:
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        api_version=None,
        # Per-LLM-call retry budget. Provider-level retries handle the common
        # transient failure modes (RateLimitError / APIConnectionError /
        # ServiceUnavailableError), specifically the Gemini 429 / RESOURCE_EXHAUSTED
        # responses that show up after sustained token throughput. The default
        # of 10 retries with up to 120s of jittered exponential backoff lets a
        # single LLM call survive a quota window that takes a couple of minutes
        # to refresh, which the previous (3, 10s) defaults could not.
        num_retries=10,
        retry_min_wait=4,
        retry_max_wait=120,
        llm_timeout=90,
        # Sampling params default to None so we send nothing to litellm and
        # the underlying model uses its provider-side default. This matters for
        # reasoning models (gpt-5, gemini-3-thinking, ...) which interact poorly
        # with explicit temperature/top_p overrides.
        llm_temperature=None,
        llm_top_p=None,
        custom_llm_provider=None,
        max_input_tokens=4096,
        # Reasoning tokens share the provider's output budget. The historical
        # 2048 cap truncated file paths and judgments with Gemini 3 Flash.
        max_output_tokens=16384,
        cost=None,
    ):

        from agent_as_a_judge.llm.cost import Cost

        self.cost = Cost()
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.llm_timeout = llm_timeout
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.num_retries = num_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.custom_llm_provider = custom_llm_provider

        self.model_info = None
        try:
            self.model_info = litellm.get_model_info(self.model_name)
        except Exception:
            print(f"Could not get model info for {self.model_name}")

        if self.max_input_tokens is None and self.model_info:
            self.max_input_tokens = self.model_info.get("max_input_tokens", 4096)
        if self.max_output_tokens is None and self.model_info:
            self.max_output_tokens = self.model_info.get("max_output_tokens", 1024)

        self._initialize_completion_function()

    def _initialize_completion_function(self):
        # Only include temperature / top_p in the partial if the caller explicitly
        # asked for them; otherwise let the underlying model use its own defaults.
        sampling_kwargs = {}
        if self.llm_temperature is not None:
            sampling_kwargs["temperature"] = self.llm_temperature
        if self.llm_top_p is not None:
            sampling_kwargs["top_p"] = self.llm_top_p

        completion_func = partial(
            litellm_completion,
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            api_version=self.api_version,
            custom_llm_provider=self.custom_llm_provider,
            max_tokens=self.max_output_tokens,
            timeout=self.llm_timeout,
            **sampling_kwargs,
        )

        def attempt_on_error(retry_state):
            print(f"Could not get model info for {self.model_name}")
            return True

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.num_retries),
            wait=wait_random_exponential(
                min=self.retry_min_wait, max=self.retry_max_wait
            ),
            retry=retry_if_exception(_retryable_provider_error),
            after=attempt_on_error,
        )
        def wrapper(*args, **kwargs):

            resp = completion_func(*args, **kwargs)
            message_back = resp["choices"][0]["message"]["content"]
            # logger.debug(message_back)
            return resp, message_back

        self._completion = wrapper

    @property
    def completion(self):
        return self._completion

    def _llm_inference(self, messages: list) -> dict:
        """Perform LLM inference using the provided messages."""
        start_time = time.time()
        response, cost, accumulated_cost = self.do_completion(
            messages=messages
        )
        inference_time = time.time() - start_time

        llm_response = response.choices[0].message["content"]
        input_token, output_token = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

        return {
            "llm_response": llm_response,
            "input_tokens": input_token,
            "output_tokens": output_token,
            "cost": cost,
            "accumulated_cost": accumulated_cost,
            "inference_time": inference_time,
        }

    def do_completion(self, *args, **kwargs):
        resp, msg = self._completion(*args, **kwargs)
        cur_cost, accumulated_cost = self.post_completion(resp)
        return resp, cur_cost, accumulated_cost

    def post_completion(self, response: str):
        try:
            cur_cost = self.completion_cost(response)
        except Exception:
            cur_cost = 0

        return cur_cost, self.cost.accumulated_cost  # , cost_msg

    def get_token_count(self, messages):
        return litellm.token_counter(model=self.model_name, messages=messages)

    def is_local(self):
        if self.base_url:
            return any(
                substring in self.base_url
                for substring in ["localhost", "127.0.0.1", "0.0.0.0"]
            )
        if self.model_name and self.model_name.startswith("ollama"):
            return True
        return False

    def completion_cost(self, response):
        if not self.is_local():
            try:
                cost = litellm_completion_cost(completion_response=response)
                if self.cost:
                    self.cost.add_cost(cost)
                return cost
            except Exception:
                print("Cost calculation not supported for this model.")
        return 0.0

    def __str__(self):
        return f"LLM(model={self.model_name}, base_url={self.base_url})"

    def __repr__(self):
        return str(self)

    def do_multimodal_completion(self, text, image_path):
        messages = self.prepare_messages(text, image_path=image_path)
        response, cur_cost, accumulated_cost = self.do_completion(messages=messages)
        return response, cur_cost, accumulated_cost

    @staticmethod
    def encode_image(image_path):
        import base64

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_messages(self, text, image_path=None):
        messages = [{"role": "user", "content": text}]
        if image_path:
            base64_image = self.encode_image(image_path)
            messages[0]["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64_image},
                },
            ]
        return messages


if __name__ == "__main__":
    load_dotenv()

    model_name = "gpt-4o-2024-08-06"
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.openai.com/v1"

    llm_instance = LLM(model=model_name, api_key=api_key, base_url=base_url)

    image_path = "/Users/zhugem/Desktop/DevAI/studio/workspace/sample/results/prediction_interactive.png"

    for i in range(1):

        multimodal_response = llm_instance.do_multimodal_completion(
            "What’s in this image?", image_path
        )
        print(multimodal_response)
