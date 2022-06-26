# Cell_segmentation
Bachelor thesis - Segmentation of cells from fluorescent microscope images

The aim of this project is to use image processing methods for segmentation of cells acquired by fluorescence microscope and provide additional information about the system.

## Project is divided to several directories
- Data
- Results
- Scripts

### Data
- directory with testing image

### Results
- directory where are created subdirectories with results of current analysis

### Scripts
- directory with scripts
  - root_script
      + creating directories for results
      + loading images to be analysed
      + calling other scripts (image_manipulation, image_analysis)
  - image_manipulation
      + provide segmented image
  - image_analysis
      + analyse segmented image
      + cooperate with shape_descriptors and export worker
  - shape_descriptors
      + compute shape descriptors
      + http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
  - export_worker
      + create charts
      + create images (boundary, centroids, ...)


## Refenrences
+ https://sisu.ut.ee/imageprocessing/book/1
+ https://www.mygreatlearning.com/blog/introduction-to-image-processing-what-is-image-processing/
+ http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
