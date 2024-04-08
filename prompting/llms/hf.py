

import time
from typing import List, Dict
import torch
import bittensor as bt

from transformers import Pipeline, pipeline, AutoTokenizer, TextIteratorStreamer
from prompting.mock import MockPipeline
from prompting.cleaners.cleaner import CleanerPipeline
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer, AutoModelForSequenceClassification
from prompting.llms import BasePipeline, BaseLLM
import logging




class CustomTextIteratorStreamer(TextIteratorStreamer):
    """
    CustomTextIteratorStreamer extends TextIteratorStreamer to add methods for checking and clearing the queue.
    """
    def has_data(self):
        """Check if the queue has data."""
        return not self.text_queue.empty()

    def clear_queue(self):
        """Clear the queue."""
        with self.text_queue.mutex:
            self.text_queue.queue.clear()



#TODO: Only change I have to figure out is how to load the math tokenizer based on the task, in my case math.



def load_hf_pipeline(
    model_id: str,
    device=None,
    tokenizer = None,
    torch_dtype=None,
    mock=False,
    _task = False,
    model_kwargs: dict = None,
    return_streamer: bool = False,
):

    """Loads the HuggingFace pipeline for the LLM, or a mock pipeline if mock=True"""
    if mock or model_id == "mock":
        return MockPipeline(model_id)

    # if not device.startswith("cuda"):
    #     bt.logging.warning("Only crazy people run this on CPU. It is not recommended.")

    if _task:
        try:
            tokenizer = tokenizer
        except Exception as e:
            bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
            raise e
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
            raise e

    streamer = CustomTextIteratorStreamer(tokenizer=tokenizer)
    
    if model_kwargs is None :
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            streamer=streamer,
        )
    else:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device_map=device,
            model_kwargs=model_kwargs,
            streamer=streamer,
        )

    if return_streamer:
        return llm_pipeline, streamer
    return llm_pipeline






class HuggingFacePipeline(BasePipeline):
    def __init__(
        self,
        model_id,
        device=None,
        tokenizer = None,
        torch_dtype=None,
        mock=False,
        model_kwargs: dict = None,
        return_streamer: bool = False,
    ):
        super().__init__()
        self.model = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.mock = mock
        self.tokenizer = tokenizer

        if tokenizer is None:
            package = load_hf_pipeline(
                model_id=model_id,
                device=device,
                torch_dtype=torch_dtype,    
                mock=mock,
                model_kwargs=model_kwargs,
            )
        else:
            package = load_hf_pipeline(
                model_id=model_id,
                tokenizer=tokenizer,
                torch_dtype=torch_dtype,
                mock=mock,
                model_kwargs=model_kwargs,
                _task = True,
            )

       

        if return_streamer:
            self.pipeline, self.streamer = package
        else:
            self.pipeline = package

        self.tokenizer = self.pipeline.tokenizer

    def __call__(self, composed_prompt: str, **kwargs: dict) -> str:
        if self.mock:
            return self.pipeline(composed_prompt, **kwargs)

        outputs = self.pipeline(composed_prompt, **kwargs)
        return outputs[0]["generated_text"]

class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):


        model_kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        super().__init__(llm_pipeline, system_prompt, model_kwargs)

        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times = [0]

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: bool = False,
        cleaner: CleanerPipeline = None,
    ):
        messages = self.messages + [{"content": message, "role": role}]

        if disregard_system_prompt:
            messages = messages[1:]

        tbeg = time.time()
        response = self.forward(messages=messages)
        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - tbeg]

        return response

    def stream(
        self,
        message: str,
        role: str = "user",
    ):
        messages = self.messages + [{"content": message, "role": role}]
        prompt = f"Question: {message}\n"


        #taskName = load_classification_pipeline(message)
#        bt.logging.info(f" \n -----------------Task Name: {taskName} --------------------------- \n")

        bt.logging.debug("Starting LLM streaming process...")
        streamer = CustomTextIteratorStreamer(tokenizer=self.llm_pipeline.tokenizer)
        _ = self.llm_pipeline(prompt, streamer=streamer, **self.model_kwargs)

        return streamer

    def __call__(self, messages: List[Dict[str, str]]):
        return self.forward(messages=messages)

    def forward(self, messages: List[Dict[str, str]]):
        composed_prompt = f"Question: {' '.join([m['content'] for m in messages])}\n"    
        
        response = self.llm_pipeline(
            composed_prompt=composed_prompt, **self.model_kwargs
        )

        response = response.replace(composed_prompt, "").strip()

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{response}"
        )
        return response

if __name__ == "__main__":
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    device = "cuda"
    torch_dtype = "float16"
    mock = True

    llm_pipeline = HuggingFacePipeline(
        model_id=model_id, device=device, torch_dtype=torch_dtype, mock=mock
    )

    llm = HuggingFaceLLM(llm_pipeline, "You are a helpful AI assistant")

    message = "What is the capital of Texas?"
    response = llm.query(message)
    print(response)


# import time
# from typing import List, Dict
# import torch
# import bittensor as bt

# from transformers import Pipeline, pipeline, AutoTokenizer, TextIteratorStreamer
# from prompting.mock import MockPipeline
# from prompting.cleaners.cleaner import CleanerPipeline
# from transformers import pipeline, TextIteratorStreamer, AutoTokenizer, AutoModelForSequenceClassification
# from prompting.llms import BasePipeline, BaseLLM
# import logging




# class CustomTextIteratorStreamer(TextIteratorStreamer):
#     """
#     CustomTextIteratorStreamer extends TextIteratorStreamer to add methods for checking and clearing the queue.
#     """
#     def has_data(self):
#         """Check if the queue has data."""
#         return not self.text_queue.empty()

#     def clear_queue(self):
#         """Clear the queue."""
#         with self.text_queue.mutex:
#             self.text_queue.queue.clear()






# def load_hf_pipeline(
#     model_id: str,
#     device=None,
#     tokenizer = None,
#     torch_dtype=None,
#     mock=False,
#     _task = False,
#     model_kwargs: dict = None,
#     return_streamer: bool = False,
# ):

#     """Loads the HuggingFace pipeline for the LLM, or a mock pipeline if mock=True"""
#     if mock or model_id == "mock":
#         return MockPipeline(model_id)

#     # if not device.startswith("cuda"):
#     #     bt.logging.warning("Only crazy people run this on CPU. It is not recommended.")

#     if _task:
#         try:
#             tokenizer = tokenizer
#         except Exception as e:
#             bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
#             raise e
#     else:
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_id)
#         except Exception as e:
#             bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
#             raise e

#     streamer = CustomTextIteratorStreamer(tokenizer=tokenizer)
    
#     if model_kwargs is None :
#         llm_pipeline = pipeline(
#             "text-generation",
#             model=model_id,
#             tokenizer=tokenizer,
#             device=device,
#             torch_dtype=torch_dtype,
#             streamer=streamer,
#         )
#     else:
#         llm_pipeline = pipeline(
#             "text-generation",
#             model=model_id,
#             tokenizer=tokenizer,
#             device_map=device,
#             model_kwargs=model_kwargs,
#             streamer=streamer,
#         )

#     if return_streamer:
#         return llm_pipeline, streamer
#     return llm_pipeline






# class HuggingFacePipeline(BasePipeline):
#     def __init__(
#         self,
#         model_id,
#         device=None,
#         tokenizer = None,
#         torch_dtype=None,
#         mock=False,
#         model_kwargs: dict = None,
#         return_streamer: bool = False,
#     ):
#         super().__init__()
#         self.model = model_id
#         self.device = device
#         self.torch_dtype = torch_dtype
#         self.mock = mock
#         self.tokenizer = tokenizer

#         if tokenizer is None:
#             package = load_hf_pipeline(
#                 model_id=model_id,
#                 device=device,
#                 torch_dtype=torch_dtype,    
#                 mock=mock,
#                 model_kwargs=model_kwargs,
#             )
#         else:
#             package = load_hf_pipeline(
#                 model_id=model_id,
#                 tokenizer=tokenizer,
#                 torch_dtype=torch_dtype,
#                 mock=mock,
#                 model_kwargs=model_kwargs,
#                 _task = True,
#             )

       

#         if return_streamer:
#             self.pipeline, self.streamer = package
#         else:
#             self.pipeline = package

#         self.tokenizer = self.pipeline.tokenizer

#     def __call__(self, composed_prompt: str, **kwargs: dict) -> str:
#         if self.mock:
#             return self.pipeline(composed_prompt, **kwargs)

#         outputs = self.pipeline(composed_prompt, **kwargs)
#         return outputs[0]["generated_text"]

# class HuggingFaceLLM(BaseLLM):
#     def __init__(
#         self,
#         llm_pipeline: BasePipeline,
#         system_prompt,
#         max_new_tokens=256,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.95,
#     ):


#         model_kwargs = dict(
#             do_sample=do_sample,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             max_new_tokens=max_new_tokens,
#         )

#         super().__init__(llm_pipeline, system_prompt, model_kwargs)

#         self.messages = [{"content": self.system_prompt, "role": "system"}]
#         self.times = [0]

#     def query(
#         self,
#         message: str,
#         role: str = "user",
#         disregard_system_prompt: bool = False,
#         cleaner: CleanerPipeline = None,
#     ):
#         messages = self.messages + [{"content": message, "role": role}]

#         #if disregard_system_prompt:
#          #   messages = messages[1:]

#         tbeg = time.time()
#         response = self.forward(messages=self.messages if not disregard_system_prompt else self.messages[1:])

#         #response = self.forward(messages=messages)
#         self.messages = messages + [{"content": response, "role": "assistant"}]
#         self.times = self.times + [0, time.time() - tbeg]

#         return response

#     def stream(
#         self,
#         message: str,
#         role: str = "user",
#     ):
#         messages = self.messages + [{"content": message, "role": role}]
#         prompt = f"Question: {message}\n"


#         #taskName = load_classification_pipeline(message)
# #        bt.logging.info(f" \n -----------------Task Name: {taskName} --------------------------- \n")

#         bt.logging.debug("Starting LLM streaming process...")
#         streamer = CustomTextIteratorStreamer(tokenizer=self.llm_pipeline.tokenizer)
#         _ = self.llm_pipeline(prompt, streamer=streamer, **self.model_kwargs)

#         return streamer

#     def __call__(self, messages: List[Dict[str, str]]):
#         return self.forward(messages=messages)

#     def forward(self, messages: List[Dict[str, str]]):
        
#         composed_prompt = f"{' '.join([m['content'] for m in messages])}\n"    
        
#         response = self.llm_pipeline(
#             composed_prompt=composed_prompt, **self.model_kwargs
#         )
        
#         split_response = response.split(composed_prompt)
        
#         if len(split_response) > 1:
#             answer = split_response[1].strip()
#         else:
#             answer = response.strip()

#         #response = response.replace(composed_prompt, "").strip()

#         bt.logging.info(
#             f"{self.__class__.__name__} generated the following output:\n{answer}"
#         )
#         return answer

# if __name__ == "__main__":
#     model_id = "HuggingFaceH4/zephyr-7b-beta"
#     device = "cuda"
#     torch_dtype = "float16"
#     mock = True

#     llm_pipeline = HuggingFacePipeline(
#         model_id=model_id, device=device, torch_dtype=torch_dtype, mock=mock
#     )

#     llm = HuggingFaceLLM(llm_pipeline, "You are a helpful AI assistant")

#     message = "What is the capital of Texas?"
#     response = llm.query(message)
#     print(response)


# # import time
# # from typing import List, Dict
# # import torch
# # import bittensor as bt
# # import logging

# # from transformers import Pipeline, pipeline, AutoTokenizer, TextIteratorStreamer
# # from prompting.mock import MockPipeline
# # from prompting.cleaners.cleaner import CleanerPipeline
# # from transformers import pipeline, TextIteratorStreamer, AutoTokenizer, AutoModelForSequenceClassification
# # from prompting.llms import BasePipeline, BaseLLM

# # logging.basicConfig(level=logging.INFO)

# # class CustomTextIteratorStreamer(TextIteratorStreamer):
# #     """
# #     CustomTextIteratorStreamer extends TextIteratorStreamer to add methods for checking and clearing the queue.
# #     """
# #     def has_data(self):
# #         """Check if the queue has data."""
# #         return not self.text_queue.empty()

# #     def clear_queue(self):
# #         """Clear the queue."""
# #         with self.text_queue.mutex:
# #             self.text_queue.queue.clear()

# # def load_hf_pipeline(
# #     model_id: str,
# #     device=None,
# #     tokenizer=None,
# #     torch_dtype=None,
# #     mock=False,
# #     _task=False,
# #     model_kwargs: dict = None,
# #     return_streamer: bool = False,
# # ):
    
# #     start_time = time.time()
# #     """Loads the HuggingFace pipeline for the LLM, or a mock pipeline if mock=True"""
# #     if mock or model_id == "mock":
# #         return MockPipeline(model_id)

# #     if _task:
# #         try:
# #             tokenizer = tokenizer
# #         except Exception as e:
# #             bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
# #             raise e
# #     else:
# #         try:
# #             tokenizer = AutoTokenizer.from_pretrained(model_id)
# #         except Exception as e:
# #             bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
# #             raise e

# #     streamer = CustomTextIteratorStreamer(tokenizer=tokenizer)
    
# #     pipeline_start_time = time.time()
# #     if model_kwargs is None:
# #         llm_pipeline = pipeline(
# #             "text-generation",
# #             model=model_id,
# #             tokenizer=tokenizer,
# #             device=device,
# #             torch_dtype=torch_dtype,
# #             streamer=streamer,
# #         )
# #     else:
# #         llm_pipeline = pipeline(
# #             "text-generation",
# #             model=model_id,
# #             tokenizer=tokenizer,
# #             device_map=device,
# #             model_kwargs=model_kwargs,
# #             streamer=streamer,
# #         )
# #     pipeline_load_time = time.time() - pipeline_start_time
# #     bt.logging.info("⏲️" * 60)
# #     bt.logging.info(f" \n Pipeline loaded in {pipeline_load_time: .4f} seconds. \n")


# #     if return_streamer:
# #         return llm_pipeline, streamer
# #     return llm_pipeline

# # class HuggingFacePipeline(BasePipeline):
# #     def __init__(
# #         self,
# #         model_id,
# #         device=None,
# #         tokenizer=None,
# #         torch_dtype=None,
# #         mock=False,
# #         model_kwargs: dict = None,
# #         return_streamer: bool = False,
# #     ):
        
        
# #         super().__init__()
# #         self.pipeline_init_start = time.time()
# #         self.model = model_id
# #         self.device = device
# #         self.torch_dtype = torch_dtype
# #         self.mock = mock
# #         self.tokenizer = tokenizer
        

# #         if tokenizer is None:
# #             package = load_hf_pipeline(
# #                 model_id=model_id,
# #                 device=device,
# #                 torch_dtype=torch_dtype,    
# #                 mock=mock,
# #                 model_kwargs=model_kwargs,
# #             )
# #         else:
# #             package = load_hf_pipeline(
# #                 model_id=model_id,
# #                 tokenizer=tokenizer,
# #                 torch_dtype=torch_dtype,
# #                 mock=mock,
# #                 model_kwargs=model_kwargs,
# #                 _task=True,
# #             )

# #         if return_streamer:
# #             self.pipeline, self.streamer = package
# #         else:
# #             self.pipeline = package

# #         self.tokenizer = self.pipeline.tokenizer
        
# #         self.pipeline_init_time = time.time() - self.pipeline_init_start
# #         bt.logging.info("⏲️" * 60)
# #         bt.logging.info(f"\n HuggingFacePipeline initialized in {self.pipeline_init_time:.4f} seconds. \n")

# #     def __call__(self, composed_prompt: str, **kwargs: dict) -> str:
        
# #         call_start_time = time.time()
# #         if self.mock:
# #             return self.pipeline(composed_prompt, **kwargs)

# #         call_time = time.time() - call_start_time
# #         outputs = self.pipeline(composed_prompt, **kwargs)
# #         bt.logging.info("⏲️" * 60)
# #         bt.logging.info(f"Pipeline call executed in {call_time:.4f} seconds.")

# #         return outputs[0]["generated_text"]

# # class HuggingFaceLLM(BaseLLM):
# #     def __init__(
# #         self,
# #         llm_pipeline: BasePipeline,
# #         system_prompt,
# #         max_new_tokens=256,
# #         do_sample=True,
# #         temperature=0.7,
# #         top_k=50,
# #         top_p=0.95,
# #     ):
# #         model_kwargs = dict(
# #             do_sample=do_sample,
# #             temperature=temperature,
# #             top_k=top_k,
# #             top_p=top_p,
# #             max_new_tokens=max_new_tokens,
# #         )

# #         super().__init__(llm_pipeline, system_prompt, model_kwargs)

# #         self.messages = [{"content": self.system_prompt, "role": "system"}]
# #         self.times = [0]

# #     def query(
# #         self,
# #         message: str,
# #         role: str = "user",
# #         disregard_system_prompt: bool = False,
# #         cleaner: CleanerPipeline = None,
        
        
# #     ):
# #         query_start_time = time.time()  
# #         messages = self.messages + [{"content": message, "role": role}]

# #         tbeg = time.time()
# #         response = self.forward(messages=self.messages if not disregard_system_prompt else self.messages[1:])
# #         query_time = time.time() - query_start_time

# #         self.messages = messages + [{"content": response, "role": "assistant"}]
# #         self.times = self.times + [0, time.time() - tbeg]
# #         bt.logging.info("⏲️" * 60)
# #         bt.logging.info(f"\n Query executed in {query_time:.4f} seconds. \n")
# #         return response

# #     def stream(
# #         self,
# #         message: str,
# #         role: str = "user",
# #     ):
        
# #         stream_start_time = time.time()
# #         messages = self.messages + [{"content": message, "role": role}]
# #         prompt = f"Question: {message}\n"

# #         bt.logging.debug("Starting LLM streaming process...")
# #         streamer = CustomTextIteratorStreamer(tokenizer=self.llm_pipeline.tokenizer)
# #         _ = self.llm_pipeline(prompt, streamer=streamer, **self.model_kwargs)

# #         stream_time = time.time() - stream_start_time
# #         bt.logging.info("⏲️" * 60)
# #         bt.logging.info(f"\n Streaming process executed in {stream_time:.4f} seconds. \n")
# #         return streamer

# #     def __call__(self, messages: List[Dict[str, str]]):
# #         call_start_time = time.time()
# #         result = self.forward(messages=messages)
# #         call_time = time.time() - call_start_time
# #         bt.logging.info("⏲️" * 60)
# #         bt.logging.info(f"\n Call executed in {call_time:.4f} seconds. \n")
# #         return result 

# #     def forward(self, messages: List[Dict[str, str]]):
# #         forward_start_time = time.time()
# #         conversation = []
# #         for message in messages:
# #             role = message['role']
# #             content = message['content']
# #             if role == 'system':
# #                 conversation.append(f"System: {content}")
# #             elif role == 'user':
# #                 conversation.append(f"User: {content}")
# #             elif role == 'assistant':
# #                 conversation.append(f"Assistant: {content}")
        
# #         composed_prompt = "\n".join(conversation) + "\nAssistant:"
        
# #         response = self.llm_pipeline(
# #             composed_prompt=composed_prompt, **self.model_kwargs
# #         )

# #         # Remove the composed_prompt from the response
# #         response = response.replace(composed_prompt, "").strip()
        
# #         forward_time = time.time() - forward_start_time
        
# #         bt.logging.info("⏲️" * 60)
# #         bt.logging.info(f"\n Forward executed in {forward_time:.4f} seconds. \n")

# #         bt.logging.info(
# #             f"{self.__class__.__name__} generated the following output:\n{response}"
# #         )
# #         return response

# # if __name__ == "__main__":
# #     model_id = "HuggingFaceH4/zephyr-7b-beta"
# #     device = "cuda"
# #     torch_dtype = "float16"
# #     mock = True

# #     llm_pipeline = HuggingFacePipeline(
# #         model_id=model_id, device=device, torch_dtype=torch_dtype, mock=mock
# #     )

# #     llm = HuggingFaceLLM(llm_pipeline, "You are a helpful AI assistant")

# #     message = "What is the capital of Texas?"
# #     response = llm.query(message)
# #     print(response)
