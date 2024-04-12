### Step 1: Install pip ###
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip

### Step 2: Clone the following repo ###
git clone https://github.com/ultralytics/yolov5

### Step 3: install the below dependency ###
cd yolov5
sudo apt install -y libfreetype6-dev 

### Step 4: Install the necessary packages ###
pip3 install -r requirements.txt

### Step 5: Clone the following repo ###
cd ~
git clone https://github.com/wang-xinyu/tensorrtx

### Step 6: copy .pt file from previous train into yolov5 directory ###

### Step 7: copy gen_wts.py from tensorrtx/yolov5 into yolov5 directory ###
cp tensorrtx/yolov5/gen_wts.py yolov5

### Step 8: Generate .wts file from .pt file, example ###
cd yolov5
python3 gen_wts.py -w best.pt -o best.wts

### Step 9: Clone the following repo ###
cd ~
git clone git@github.com:caoducanh11320000/trt-inference-yolov5.git

### Step 10: Open config.h ###
cd trt-inference-yolov5/include 
vi config.h

### Step 11: Change kNumClass to number of class your model is trained ###
Exemple:
kNumClass = 9;

### Step 12: Create build directory ###
mkdir build 
cd build

### Step 13: Copy .wts file which is generate in step 7 into this build directory ###
Example:
cp ~/yolov5/best.wts .

### Step 14: Compile it ###
cmake ..
make

### Step 15: Serialize model ###
./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]
Example
./yolov5_det -s best.wts best.engine n6

### Step 16: Deserialize model ###
**With image, images is folder contain images**
./yolov5_det -d -i best.engine images

**With video**
./yolov5_det -d -v [.engine] [path to video]
Example
./yolov5_det -d -v best.engine ../video/test_1.mp4

