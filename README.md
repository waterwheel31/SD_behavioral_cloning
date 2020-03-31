# Behavioral Cloning


[![Youtube](http://img.youtube.com/vi/35PKVw-BydI/0.jpg)](https://www.youtube.com/watch?v=35PKVw-BydI "Pathplanning")
  (click to see Youtube Video)

## Objective 

This tries to let a driving agent learn how to drive by seeing human's drive records. 


## Approach

- Record manually drived virtual vehicle
    - Used a driving course environment on Unity
    - drived a virtual car, and recorded followings:
        -  images of from the vehicle from 3 points 
            -  front(center)
            -  front(biased-to-left)
            -  front(biased-toright))
        -  sterring input by human

    <p align="center">
    <img src="./images/human_drive.png" width=300>

- Trained CNN model 
    - Used the images as the input and the steering movement as the output
    - Recorded 4 laps of a same environement. Recorded 50+ times at difficult to learn places (in front of the dirt area and steep curb in the end)
    - Used following CNN network. This is based on (but modified) NVIDA's network for self-driving car  (https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

    <p align="center">
    <img src="./images/network.png" width = 300>
    </p>


    - Data is preprrocessed as below:
        - Cropped the top and bottom of the images and focused on the area where the road is shown 
        - Resized to (200 x 66) 
        - normalized the data between -0.5 - +0.5 
    - Data is Augumented and increased when training
        - add left-right flipped images (+ flipped sterring)
        - image from left camera and right camera (as mentioned above). Adjusted the right camera image's corresponding steering as -0.2 (toward left) and the left camera's images as +0.2 (toward right) 
        - historam equalization
        - randomly adjusted the brightness (to avoide the impact of shadow) 
    
## Result 

- The car can run one wrap in the same course 
- See the video above 

## Limiation and improvement opportunities
- This depends on images and does not understand where is the roads. Therefore, when the car faces unseen scenary (ex.out of the road), the car losts control. This can be solved by adding the concept of localization.  
- Currently, the output is only limited to sterring and the slottle value is fixed. By adding slottle motions as the output, the drive will be more natural, thought more training will be required

## How to Run 

- training the model 'python train.py'
- run the model 'python drive.py model.h5'
- run Unity simulator (the environment is on Udacity server. The way to run localy is ... (TBD))