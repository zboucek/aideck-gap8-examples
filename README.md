# AI-deck examples repository

Check out the [documentation](https://www.bitcraze.io/documentation/repository/aideck-gap8-examples/master/)
for starting guides. 


[Getting started with the AI deck | Bitcraze](https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/)

[Documentation for AI deck examples | Bitcraze](https://www.bitcraze.io/documentation/repository/aideck-gap8-examples/master/) 

[Classification Demo | Bitcraze](https://www.bitcraze.io/documentation/repository/aideck-gap8-examples/master/ai-examples/classification-demo/)

```bash
docker run --rm -it --name myAiDeckContainer aideck-with-autotiler
```

If firmware upload via docker fails, it needs to be uploaded wirelessly via crazyradio with already connected AI Deck (it also contains firmware).

# Crazyflie drones and AP addresses

### Drones:

usb://0

radio://0/100/2M/E7E7E7E701

- facedetection

radio://0/100/2M/E7E7E7E704 

- classification

radio 01-04

### AP

192.168.100.1

SSID: kky-un509

# CF firmware with configured WiFi

```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
cd crazyflie-firmware
```

KBuild allows setting parameters for custom build (e.g. that AI deck does not create AP, but connects to already existing one. SSID and password need to be set). 

```bash 
make menuconfig
```

```bash
make
cfloader flash cf2.bin stm32-fw -w radio://0/100/2M/E7E7E7E70X
```

# Update ESC firmware (WiFi)

It is necessary for NINA module to connect to AP properly and for CF IP address to be reported in cfclient log.

```bash
git clone https://github.com/bitcraze/aideck-esp-firmware
cd aideck-esp-firmware
docker run --rm -it -v $PWD:/module/ --device /dev/ttyUSB0 --privileged -P bitcraze/builder /bin/bash -c "make" 
cfloader flash build/aideck_esp.bin deck-bcAI:esp-fw -w radio://0/100/2M/E7E7E7E70X
```

But it can also be uploaded via JTAG !!!use pins for NINA, not for GAP8)

# Old version of AI Deck GAP8 bootloader (uploading via CFRadio gets stuck at 4 or 99%)

Firmware upload for GAP8 via crazyradio will not work properly, bootloader needs to be uploaded via JTAG:

```bash
git clone https://github.com/bitcraze/aideck-gap8-bootloader.git
cd aideck-gap8-bootloader
docker run --rm -it -v $PWD:/module/ --device /dev/ttyUSB0 --privileged -P bitcraze/aideck /bin/bash -c 'export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg; source /gap_sdk/configs/ai_deck.sh; cd /module/; make all image flash'

```

# WiFi Video Streamer 

X) *opencv* needs to be installed

```bash
pip install opencv-python
```

1) upload demo

```bash
cd ~/git/aideck-gap8-examples
docker run --rm -v ${PWD}:/module aideck-with-autotiler tools/build/make-example examples/other/wifi-img-streamer image
cfloader flash examples/other/wifi-img-streamer/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img deck-bcAI:gap8-fw -w radio://0/100/2M/E7E7E7E70X
```

2) start client

```bash
cd ~/git/aideck-gap8-examples/examples/other/wifi-img-streamer  
python opencv-viewer.py -n 192.168.100.XXX
```

for saving:

```bash
python opencv-viewer.py --save -n 192.168.100.XXX
```

# Face Detection Demo

upload demo: 

```bash
cd ~/git/aideck-gap8-examples
docker run --rm -v ${PWD}:/module aideck-with-autotiler tools/build/make-example examples/image_processing/FaceDetection clean model build image
cfloader flash examples/image_processing/FaceDetection/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img deck-bcAI:gap8-fw -w radio://0/100/2M/E7E7E7E70X
```

start client:

```bash  
cd ~/git/aideck-gap8-examples/examples/image_processing/FaceDetection
python3 opencv-viewer.py -n 192.168.100.XXX
```

# Classification Demo

## Preparation

Install tensorflow environment either in anaconda, miniconda or virtualenv:

```bash  
conda create -n aideck python=3.10.9
conda activate aideck
cd ~/git/aideck-gap8-examples/examples/ai/classification/
pip install -r requirements.txt
```

## Prepare dataset  

Put grayscale images for individual classes into `~/git/aideck-gap8-examples/examples/ai/classification/images` folder. Each folder will be named after the class.

Run `split_data.py`: it will split data from folders in images to training and testing folders in training_data as it should be in the example: 

```bash
/train/class_1/*.jpeg
/train/class_2/*.jpeg  
/validation/class_1/*.jpeg
/validation/class_2/*.jpeg
```

Edit Dense layer in `train_classifier.py` 

- `tf.keras.layers.Dense(units=2` where 2 should be replaced by number of classes)

## Start training

1) Run `train_classifier.py` with edited dense layer according to number of classes (this will run with default parameters, can be seen in `parse_args()` function)

```bash
cd ~/git/aideck-gap8-examples/examples/ai/classification/  
python train_classifier.py
```
for 50 epochs:

```bash
cd ~/git/aideck-gap8-examples/examples/ai/classification/
python train_classifier.py --epochs 50 
```

2) Edit `classification.c` so that print reflects current classes: edit `static const char* classes[]` content  

## Compile and upload via crazyradio

```bash
cd ~/git/aideck-gap8-examples/
docker run --rm -v ${PWD}:/module aideck-with-autotiler tools/build/make-example examples/ai/classification clean model build image
cfloader flash examples/ai/classification/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img deck-bcAI:gap8-fw -w radio://0/100/2M/E7E7E7E70X
```
