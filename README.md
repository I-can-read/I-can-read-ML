# I-can-read-ML

## Preprocess Image
In order to recognize text in an image well, it is almost essential to go through image preprocessing.
I performed image preprocessing through the following process.

1. Change to gray
2. Resize image
3. Remove noise
4. Use Adaptive Thresholding

After such image preprocessing, a text-extracting model was created.

</br>

## Try1. Using Tesseract
I simply imported tesseract from the library and extracted the text. However, it did not perform well for Korean language. Here's how to use it. (I did not want to extract numbers from the image, so I specified the blacklist as follows.)

``` Python
print(tesseract.image_to_string(Image.open(result), config=r'-c tessedit_char_blacklist=0123456789 --psm 3', lang='kor'))
```

</br>

## Try2. Using Tesseract4 with tessdata_best
There was training data for the specific language of Tesseract itself. So, I tried text extraction using this Korean data that showed the best performance. Although the recognition rate was higher than before, the performance was still not good. Finally I decided that I should train Korean language myself. The picture below is the additional value I put in to better recognize the cafe menu items.

<img width="559" alt="1" src="https://user-images.githubusercontent.com/74898231/229517970-46d5c4a0-95aa-4bba-ac4a-c1477ae577bf.png">


</br>

## Try3. Using Clova ai model
To better recognize Korean language, it seemed best to use a model made by Koreans. And I had to use a model that could train without difficulty. So, I decided to train the model using the [deep-text-recognition-benchmark created by Clova ai](https://github.com/clovaai/deep-text-recognition-benchmark). And I used [Korean font images from aihub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100) to train the data. This data consists of 50 types of fonts using 11,172 modern Korean characters, image files created by gender and age group, and images constructed with 100,000 images including signs, trademarks, and traffic signs.

* Create the lmdb data
  ```
  python3 ./deep-text-recognition-benchmark/create_lmdb_dataset.py \
    --inputPath ./deep-text-recognition-benchmark/ocr_data/ \
    --gtFile ./deep-text-recognition-benchmark/ocr_data/gt_train.txt \
    --outputPath ./deep-text-recognition-benchmark/ocr_data_lmdb/train
  ```

  ```
  python3 ./deep-text-recognition-benchmark/create_lmdb_dataset.py \
    --inputPath ./deep-text-recognition-benchmark/ocr_data/ \
    --gtFile ./deep-text-recognition-benchmark/ocr_data/gt_validation.txt \
    --outputPath ./deep-text-recognition-benchmark/ocr_data_lmdb/validation
  ```

</br>

* Train the model
  ```
  CUDA_VISIBLE_DEVICES=0 python3 ./deep_text_recognition_benchmark/train.py \
      --train_data ./deep_text_recognition_benchmark/ocr_data_lmdb/train \
      --valid_data ./deep_text_recognition_benchmark/ocr_data_lmdb/validation \
      --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
      --batch_size 512 --batch_max_length 200 --data_filtering_off --workers 0 \
      --num_iter 100000 --valInterval 100
  ```

</br>

* Result

  The best accuarcy is 99.031.
  <img src="https://user-images.githubusercontent.com/74898231/229519460-779bc30e-50d6-4ae5-b5d2-9330cdb08e37.png">
  <img src="https://user-images.githubusercontent.com/74898231/229519799-3cefa140-253b-415c-b492-1c7b30f06c1e.png">


</br>

## Try4. Train cafe menu items with my own data
A model with very high performance was created, but I wanted to create a model that better recognizes cafe menu items. So, I created my own dataset of cafe menu items and fine-tuned the existing model with this data. This is the final model I made.
