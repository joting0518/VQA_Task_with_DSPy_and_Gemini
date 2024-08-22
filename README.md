# VQA Task Using DSPy and Gemini

This project aims to utilize DSPy for Visual Question Answering (VQA) tasks. The goal is to leverage DSPy to generate useful prompts that aid the Visual Language Model (VLM), specifically Gemini, in solving questions based on video content.

## Project Overview

The experiment involves the following steps:

1. **Understanding DSPy and Modifying Code for VQA**  
   Begin with a basic understanding of DSPy and make the necessary modifications to adapt it to the VQA task.

2. **Custom Class Implementation for Gemini-1.5-pro**  
   Create a custom class to replace the original `dspy.google` package, allowing the use of Gemini-1.5-pro instead of Gemini-1.5-flash.

3. **Handling Deep-Copy Issues**  
   Address the deep-copy problem, where DSPy attempts to deep copy the entity. Since there is an inference entity, deep-copying leads to errors, which need to be resolved.

4. **Incorporating CoT and General Prediction**  
   Explore the integration of Chain of Thought (CoT) reasoning and general predictions. DSPy does not modify prompts directly but proves helpful in generating useful thoughts to solve the questions.
