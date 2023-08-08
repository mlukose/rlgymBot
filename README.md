# rlgymBot
# Description
- To Be Added

# How to Run
- Must have both rocket leage (either steam or epic games version) as well as bakkes mod installed.
- Open bakkesmod but do NOT open rocket league.
- Run the rlbot.py file with an argument specifying either "training" or "predicting". This should open the game and start everything automatically.
```{bash}
python rlbot.py training
```
- Before running make sure to update the filepath variable above the main loop (found at bottom of the file) to change where the model is saved to. This path is also where the prediction loop will find the model you wish to use.
