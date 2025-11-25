# 491 Assignment: Policy vs. Value Exploration

To minimize the amount of times each program would need to be run, I configured reinforce_lunar and sarsa_lunar to read 8 different global-config files each of which coincide with a configuration variation stated in the rubic.

To run each algorithm:
1. Create a python vm and source it
2. pip install -r requirements.txt
3. run script: python3 {NAME OF FILE}.py

### Structure of Output
After each program has finished running, the results are saved in the ./results directory. In there you can see the output is then split up into "reinforce" and "sarsa." Then each directory inside is named after the config file that was used for generate that output (i.e global_config-1-g-90 would be the config file of the same name where the only variation is that gamma was set to 90). 
Inside each of these folders you'll see 4 output files:

episode_rewards.txt - The resulting reward from each episode.
global_config-... - copy of the config file used
learning_curve.png - the mean and standard deviation plot from the the full 3000 episode run with 20-Episode windows.
summary.txt - contains important information such as training time, average reward, std, max and min rewards across the entire 3000 episode run.
training_plot.py - 2 plots, left one is the total reward plotted for every episode. Right one is the 20-Episode Moving Average.

NOTE: The results currently in the repo were produced using device = cpu on a Macbook Pro 2021 M1 Pro
