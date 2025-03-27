from setuptools import setup, find_packages

setup(
    name="vrp_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pytest>=6.2.5',
        'matplotlib>=3.4.3',
        'typing-extensions>=4.0.0',
        'flask>=2.0.0',
        'tabulate>=0.8.0'
    ],
    author="VRP team",
    author_email="vanquoc11082004@gmaill.com",
    description="Vehicle Routing Problem System",
    keywords="vrp, routing, optimization",
    python_requires='>=3.8',
)