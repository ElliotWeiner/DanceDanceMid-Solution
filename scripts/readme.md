HOW TO USE DATA ANNOTATION TOOL:

python3 annotation_chunking.py <FILENAME WITHOUT PATH>

Specificed mp4 file should be in dataset/images1



Data will automatically be chunked if ran from scripts folder and mp4 filename is correct.

Annotation gui will pop up playing 4 views of a 3-image-gif player view


Annotations should reflext the end-frame of the gif (i.e. if the players feet are placed on the correct areas).

Left foot:
  q - up (front)
  w - down (back)
  e - left
  r - right
  t - none

Right foot:
  a - up (front)
  s - down (back)
  d - left
  f - right
  g - none

space - next gif
b - previous gif

Data will only save when all gifs have been classified
