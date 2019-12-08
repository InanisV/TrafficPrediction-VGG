# TrafficPrediction-VGG
Traffic Flow Prediction with Convolutional Neural Networks (VGG16)

## Brief Intro

This implementation predicts the speed of cars in the next five minutes based on past two hours.

The test database is **[PeMSD7](http://pems.dot.ca.gov/)**, a database collected from Caltrans Performance Measurement System (PeMS) in real-time by over 39, 000 sensor stations.

*To notice that VGG only takes temporal information into consideration.*

## Usage Explanation

Dataset.py
```python
dataset(data_v, time_slot, predict_slot, batch_size)
# data_v        -- the input metrix with road_num x time
# time_slot     -- the number of time slots used for prediction
# predict_slot  -- the number of time slots to predict
# batch_size    -- batch size
```

VGG16.py
```python
VGG16(num_classes)
# num_classes   -- the number of roads to predict
```

VGG_main.py
