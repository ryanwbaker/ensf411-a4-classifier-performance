# lab4: Finger counting in production
## Ultimate Goal
 In the coming three labs, we would like to build a Tkinter GUI application that can recognize finger counting for digits from 1 to 5. The final applictation will capture images form a webcam and display the finger count as a string overlay. We will proceed in 3 steps:
 1. Lab2: Data acquisition from webcam: Build a GUI application to acquire image data.
 1. Lab3: Train a fastai learner to recognize the 5 finger counts.
 1. Lab4: Wrap the model in a GUI application to predict digits from live webcam stream.

 We will implement the [American counting system](https://en.wikipedia.org/wiki/Finger-counting#Western_world)
 
## Lab4 Goal
 
Assess model performance in production environment. The script `lab4_predict_finger_count_gui.py` provides a GUI with the capability to load a fastai learner, by default `models/finger_count.pkl`, and to add predicted label and probability as a string overlay to the video feed. Your task is to evaluate your lab3 model performance in this *production environment*.

Here are three possible ways to assess production performance:

**1. Save labeled screenshots**

Add code to `lab4_predict_finger_count_gui.py` that allows saving screenshots of labeled images. As the expert, you will review these data.

**2. Print labels to standard out**

A start marker is printed followed by the predicted digit to standard out, and a stop marker at the end. The idea is that after starting the experiment, a unique digit is shown to the model, varying the position of the hand. The output should match the chosen finger digit. This is repeated for all five digits.

**3. Acquire a test dataset**

Acquire a test data set, use your lab3 model to generate predictions.

Choose one of these aproaches and save the resulting data in `lab4-results.csv`. Analysis of these results is completed in `lab4_evaluate_finger_count.ipynb` and includes computation of accuracy and confusion matrix. There should be at least 5 examples for each digit.

## Optional tasks
To speed up video feed, put prediction in separate Python process. Use `multiprocessing.Process` and `multiprocessing.Queue`.

## What to hand in
- Select one of the assessment strategies.
- Any code to facilitate your assessment strategy
- Analyze frames and add results to `lab4-results.csv`
- In the Jupyter notebook `lab4_evaluate_finger_count.ipynb`:
  - calculate accuracy and confusion matrix for data in `lab4-results.csv`
  - answer questions
  - complete *Summary and Conclusion*
  - complete *Reflection*
- Keep code clean, comment/document and remove any unnecessary cells in the notebook.

During development, save progress with git and use descriptive commit messages.

Hand in: git push `lab4_predict_finger_count_gui.py`, `lab4_evaluate_finger_count.ipynb`, `lab4-results.csv`, verify on github, submit url on D2L.

**Important:** Do **not** commit image data to github. Images do **not** need to be handed in.
          