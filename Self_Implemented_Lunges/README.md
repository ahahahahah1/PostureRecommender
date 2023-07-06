1. Chose files for test and train. Multiple people are being used to train but only Anirudh's videos will be used for testing.
2. To install mediapipe, follow the instructions at https://developers.google.com/mediapipe/framework/getting_started/install. I was installing it on WSL. Due to OpenCV version 4 and Ubuntu 22.04, there were some changes that have to be made. Additionally, instead of libdc1394-22-dev in ___, need to replace it with libdc1394-dev. 
3. Didn't install Windows ADB as I don't think it is necessary.

---
Simply running `pip3 install mediapipe` solved this issue.
I had divided my input video files into test and train folders. Had to make some changes in the scripts to accomodate that change in the directory structure.

Faced this error
> This application failed to start because no Qt platform plugin could be initialized.

Localized the error to the command `cv2.imshow()`. I simply commented it out for now since couldn't find an immediate fix online.


### Temporarily setup SSH key on WSL as well. Will need to later delete that and setup sharing of SSH keys between WSL and Windows.