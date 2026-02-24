<h2>Initial draft, subject to continuous updates during the review process.</h1>

<h1>Installation</h1>
<h2>Requirement</h2>
<ul>
<li>Linux with Python >=3.8</li>
<li>Matlab >=2020a</li>
</ul>
<h2>Example conda environment setup</h2>
conda create --name TFD python=3.8 -y <br>
conda activate TFD<br>
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge<br>

<h3>Mask2Former is required</h3>
git clone git@github.com:facebookresearch/detectron2.git<br>
cd detectron2<br>
pip install -e .<br>
pip install git+https://github.com/cocodataset/panopticapi.git<br>
pip install git+https://github.com/mcordts/cityscapesScripts.git<br>
cd ..<br>
git clone git@github.com:facebookresearch/Mask2Former.git<br>
cd Mask2Former<br>
cd mask2former/modeling/pixel_decoder/ops<br>
sh make.sh<br>

<h3>Other requirements </h3>
pip install flask fvcore scipy opencv-python<br>

<h1>Training</h1>
python stage1/complexTrain.py
python stage2/complexTrain.py

<h1>Test</h1>
<h3>Run TFD-Net as a service</h3>
In TFNet_server.py Line 23: Change to your IP. <br>
python test/TFD_exe/TFNet_server.py <br>
<h3>Matlab code: Generate test data</h3>
main function entrance: gen_dataset_complex.m <br>
<h3>Matlab code: Test TFD-Net</h3>
main function entrance: main_TFD.m


