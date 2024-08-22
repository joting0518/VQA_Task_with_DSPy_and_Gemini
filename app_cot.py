import dspy
import os
from collections.abc import Iterable
from typing import Any, Optional
import backoff
from dsp.modules.lm import LM
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError
import time
from dspy.teleprompt import BootstrapFewShot
import pandas as pd
import csv

# 覆寫 dspy.google 的功能，改成使用 1.5-pro 並增加 upload 的功能
try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPICallError
    google_api_error = GoogleAPICallError
except ImportError:
    google_api_error = Exception
    # print("Not loading Google because it is not installed.")

def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    # answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    # return answer_EM and answer_PM
    return answer_EM

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


def giveup_hdlr(details):
    """Wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True


BLOCK_ONLY_HIGH = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
]


class CustomGoogle(dspy.Google):
    """Custom wrapper around Google's API, extending dspy.Google."""

    def __init__(
        self,
        model: str = "models/gemini-1.5-pro",
        api_key: Optional[str] = None,
        safety_settings: Optional[Iterable] = BLOCK_ONLY_HIGH,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Google to use?
        api_key : str
            The API key for Google.
            It can be obtained from https://cloud.google.com/generative-ai-studio
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model=model, api_key=api_key, safety_settings=safety_settings, **kwargs)
        
        # Additional initialization for CustomGoogle if needed
        self.custom_feature = kwargs.get("custom_feature", None)

    def upload_video(self, path: str, display_name: str) -> dict:
        """Upload a video file to Google API and return the file info."""
        sample_file = genai.upload_file(path=path, display_name=display_name)
        print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
        return sample_file
    
    def basic_request(self, prompt: str, **kwargs):
        """Override basic_request to add custom behavior."""
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            **kwargs,
        }

        # Google disallows "n" arguments
        n = kwargs.pop("n", None)
        if n is not None and n > 1 and kwargs['temperature'] == 0.0:
            kwargs['temperature'] = 0.7

        response = self.llm.generate_content(prompt, generation_config=kwargs)

        # Log or modify the response if needed
        history = {
            "prompt": prompt,
            "response": [response],
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        # Here you can add any custom post-processing on the response
        if self.custom_feature:
            print(f"Custom feature enabled: {self.custom_feature}")
            # Apply custom feature logic here if needed

        return response

    @backoff.on_exception(
        backoff.expo,
        (google_api_error),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Google whilst handling API errors."""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        """Override __call__ to add custom behavior."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)

        completions = []
        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.parts[0].text)

        return completions


data = []
with open('/Users/chenruoting/Desktop/prompt_auto_adjust/video_and_question_val.csv', mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  
    for row in reader:
        data.append(row)

with open('/Users/chenruoting/Desktop/prompt_auto_adjust/val_output.csv', mode='w', newline='', encoding='utf-8') as outfile:
    
    writer = csv.writer(outfile)
    writer.writerow(['video','response','question_id'])
    # writer.writerow(['video','response1','response2'])
    for row in data:
        print(row)
        video_id, question, question_id = row
        full_video_path = os.path.join('/Users/chenruoting/Desktop/videos_val/', video_id+'.mp4')


        # 配置 dspy 使用 Google 的 Gemini API
        # 修改 env/lib/python3.10/site-packages/dsp/modules/google.py 裡面的 default 功能
        gemini = CustomGoogle("models/gemini-1.5-pro", api_key="replace your api key")
        dspy.settings.configure(lm=gemini, max_tokens=1024)

        class BasicQA(dspy.Signature):
            """Answer questions with short factoid answers."""
            question = dspy.InputField()
            video = dspy.InputField()
            instruction = dspy.InputField()
            answer = dspy.OutputField(desc="often between 1 and 5 words")


        class ZeroShot(dspy.Module):
            def __init__(self):
                super().__init__()
                self.prog = dspy.ChainOfThought(BasicQA)
                self.google_api = gemini

            def __deepcopy__(self, memo):
                # 防止 deepcopy 複製 google_api 類
                new_instance = ZeroShot()
                new_instance.prog = self.prog
                # 不複製 google_api 類
                new_instance.google_api = self.google_api
                return new_instance
            
            def forward(self, video, question, instruction):
                uploaded_file = self.google_api.upload_video(path=video, display_name="Sample Video")
                time.sleep(5)
                # pred = self.prog(
                #     question=question, 
                #     video=uploaded_file.uri, 
                #     instruction=instruction
                # )
                pred = self.prog(
                    question=question, 
                    video=uploaded_file.uri,
                    instruction=instruction
                )
                
                return pred

        time.sleep(5)
        zero_shot_instance = ZeroShot()

        video_path = full_video_path
        question = question
        instruction = "Please specify the objects based on the question, such as color, position(right, middle, left), shape."
        
        # add training set
        """
        trainset = [
            dspy.Example(
                question="From the containers placed on the table by the person, in which ones could you pour liquid without spilling?", 
                video="/Users/chenruoting/Desktop/prompt_auto_adjust/video_7884.mp4",
                answer="red cup, yellow cup"
            ).with_inputs("question", "video"),
            dspy.Example(
                question="From the containers that the person pours into, which one is the widest?", 
                video="/Users/chenruoting/Desktop/prompt_auto_adjust/video_825.mp4",
                answer="The middle jar",
                instruction=instruction
            ).with_inputs("question", "video","instruction"),
            dspy.Example(
                question="From the containers that the person pours into, which one is the tallest?", 
                video="/Users/chenruoting/Desktop/prompt_auto_adjust/video_10943.mp4",
                answer="The leftest glass-jar",
                instruction=instruction
            ).with_inputs("question", "video","instruction"),
        ]
        """
        # Please specify the objects based on the question, such as color, position(right, middle, left), shape. 
        answer = zero_shot_instance(video=video_path, question=question, instruction=instruction)
        print(f"rationale: {answer.rationale}")
        print(f"Answer: {answer.answer}")
        writer.writerow([full_video_path,answer.answer,question_id])
        print("-----------------------------------------------------------------")

        # compile model and print results in every iteration
        """
        # Set up a basic teleprompter, which will compile our RAG program.
        teleprompter = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)

        for i in range(3):
            print(f"Iteration {i+1}:")
            student_model = ZeroShot()
            teacher_model = ZeroShot()

            compiled = teleprompter.compile(student_model, teacher=teacher_model, trainset=trainset)
            pred = compiled(question=question, video=video_path, instruction=instruction)
            # pred = compiled(question=question, video=video_path)
            print(f"Compiled Answer {i+1}: {pred}\n")
            print("History: ")
            gemini.inspect_history(n=1)
        """
       

        # 如果 data 有相同的問題，那預測後的結果會直接複製該答案
        # 若沒有的話，會返回籠統回應，如：The two cups
        # 在 loop 會誤讀影片，應 7884.mp4 卻讀成 example 的 825.mp4(last trainset)
