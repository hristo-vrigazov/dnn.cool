from setuptools import setup, find_packages
import pathlib
import dnn_cool

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="dnn_cool",
    version=dnn_cool.__version__,
    description="DNN.Cool: Multi-task learning for Deep Neural Networks (DNN).",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hristo-vrigazov/dnn.cool",
    author="Hristo Vrigazov",
    author_email="hvrigazov@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["torch",
                      "pandas",
                      "numpy",
                      "tqdm",
                      "treelib",
                      "joblib"],
    extras_require={
        'catalyst': ["catalyst == 20.12", "scikit_learn", "matplotlib"],
        'nlp': ['transformers'],
        'interpretability': ['captum']
    }
)
