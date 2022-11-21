# ITU-ML5G-PS-005-KDDI-UT-NakaoLab-AI

# Environment
linux ubuntu 20.04 LTS<br>
python = 3.9.13

# Getting Started
1. Create virtual environment
   
    If you use anaconda:

    ```
    $ conda create -n itu python=3.9.13
    $ conda activate itu
    ```

    If you use venv:
  
    ```
    $ sudo apt install -y python3-venv
    $ python3 -m venv ~/python/itu
    $ source ~/python/itu/bin/activate
    ```

2. Switch to project directory
   
   ```
   $ cd <your path>/repository name
   ```
3. Install libraries
   
   ```
   $ pip install -r requirements.txt
   ```

4. Install tensorflow
   
   If you use M1 mac:
  
   ```
   $ conda install -c apple tensorflow-deps==2.9.0
   $ pip install tensorflow-macos==2.9.0
   $ pip install tensorflow-metal
   ```

   else:

   ```
   $ pip install tensorflow==2.9.0
   ```
