# Bird Classification Using OpenCV 

# Project Overview

This project utilizes Computer Vision techniques to increase the quality and quantity of data collected in the conservation field. If successful, it will reduce the burden on developing countries to accurately document wildlife-related data, as well as aid in data collection for geographically challenging (such as the Arctic) or expansive areas (such as in a migration pattern from North America to South America). Further extensions of this project can focus on sorting species based on video data.

The project will attempt to find a solution to the following: 

"How might we use Machine Learning to increase the accuracy of bird species classification based on images"

Computer Vision (CV) is a field of artificial intelligence that trains computers to interpret and understand the visual world. A properly trained model can accurately identify and classify large quantities of images. 

This has many positive implications for CV in the wildlife conservation field, such as helping researchers to:
* gather data effectively on population and geographical distribution of species
* track migratory patterns and impacts of ecological changes
* study behaviour collectively and make strategies to conserve species

[Source](https://aiworldschool.com/research/this-is-why-ai-in-wildlife-conservation-is-so-glorious/)

This project relies heavily on the interpretation of the relative difference or similarity of bird images across different species. 
Some [background information](https://www.allaboutcircuits.com/technical-articles/image-histogram-characteristics-machine-learning-image-processing/) on image histograms may be required. 

# Project Motivation

A 2022 [study](https://news.cornell.edu/stories/2022/05/global-bird-populations-steadily-decline) revealed that 48% of existing bird species worldwide are known or suspected to be undergoing population declines. Populations are only stable for 39% of species. Only 6% are showing increasing population trends, and the status of 7% is still unknown. Birds are important ecological indicators that are critical to many environmental monitoring schemes, bidiversity assessments and conservation decision-making.

Common reasons for the decline in bird populations include agricultural activity, urban development, natural resource extraction, chemical pesticides, and industrial contaminants.  Loss of bird habitat affects both terrestrial and aquatic environments, including marine areas. 

According to conservation scientists, [current data collection approaches are not adequate](https://sekercioglu.biology.utah.edu/PDFs/2020%20Monitoring%20the%20world's%20bird%20population%20with%20community%20science%20data.pdf) for monitoring species across geographic ranges can be difficult and resource intensive, and rely too heavily on community data, which may be lacking in both data quantity and quality. 

In a few countries, birds are monitored using government-coordinated surveys that produce reliable national-level population trends. However, formal surveys such as these are often lacking in developing nations due to the resources required. This is especially concerning because these regions harbor the majority of the world's bird species. 



## Data set

This project uses this [dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species), which consists of 84635 training images, 2625 test images, and 2625 validation images across 525 bird species. There are no null or duplicate values. The images were cropped so that the bird in most cases occupies at least 50% of the pixel in the image. Then the images were resized to 224 X 224 X3 in jpg format. Each species has at least 130 training images, with some species having more.

## Data Dictionary

<table>
  <tr>
    <th style="text-align: left; background: lightgrey">Column Name</th>
    <th style="text-align: left; background: lightgrey">Description</th>
  </tr>
  <tr>
    <td style="text-align: left"> <code>labels</code> </td>
    <td style="text-align: left">bird species associated with the image file</td>
  </tr>
    <tr>
    <td style="text-align: left"><code>scientific label</code></td>
    <td style="text-align: left">scientific name for the bird species</td>
  </tr>
  <tr>
    <td style="text-align: left"><code>filepaths</code></td>
    <td style="text-align: left">the relative file path to an image file</td>
  </tr>
    <tr>
    <td style="text-align: left"><code>data set</code></td>
    <td style="text-align: left">which dataset (train, test or valid) the image filepath belongs to</td>
  </tr>
      <tr>
    <td style="text-align: left"><code>data set</code></td>
    <td style="text-align: left">which dataset (train, test or valid) the image filepath belongs to</td>
  </tr>
       <tr>
    <td style="text-align: left"><code>class_id</code></td>
    <td style="text-align: left">the class index value associated with the image file's class</td>
  </tr>

</table>

### Dependencies

* This project requires opencv-python to run
```
pip install opencv-python
```

It also requires the following dependencies and imports

`````````
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import plotly.express as px
import cv2
from IPython.display import Image, display
`````````

## Author

Larissa Huang


## Acknowledgments

Inspiration, code snippets, etc.
* [Medium article by Raghunath D about OpenCV image histograms](https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7)
