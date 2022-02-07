from setuptools import setup, find_packages

setup(
    name='videocls',
    version='0.0.1',
    description='Simple Video classification and Action Recognition',
    url='https://github.com/sithu31296/video-classification',
    author='Sithu Aung',
    author_email='sithu31296@gmail.com',
    license='MIT',
    packages=find_packages(include=['videocls']),
    install_requires=[
        'tqdm',
        'tabulate',
        'pytorchvideo',
        'yt-dlp',
        'joblib',
    ]
)