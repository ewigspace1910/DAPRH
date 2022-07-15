from setuptools import setup, find_packages


setup(name='Ewig@S.P-MixFramework',
      version='1.0.0',
      description='Combine both P2LR and PPLR framwork ideal',
      author='EwigSpace1910',
      author_email='ducanhng.work@gmail.com',
      url='https://github.com/ewigspace1910/ReID_maket1501/tree/main/_MixFrameworks',
      install_requires=[
          'numpy', 'torch', 'torchvision', 'tensorboard',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'scikit-image', 'seaborn', 'pandas',
          'metric-learn', 'faiss_gpu==1.6.3', 'gdown'],

      packages=find_packages()
      )