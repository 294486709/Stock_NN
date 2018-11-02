# Stock NN  
## Project Description  
The project is about using historical data and NN to predict the trend of a stock.  
## Appraoch 
First, we download stock data with stock symbol, data, open, close, high, low and volumn. Then we filter out stock with too low price and volumn.
After, we find functions to calculate up to 20 differnet stock indicators. Then we use historical data to trian the model and try
to find out the best stock indicator for the project. Among the 20 indicators, we choose 5 of the them with the best accuracy.
 Then we tuned the model and tried to find the best prarmeter for the model. Also, we classify our result into five different sets(<-3%,-3%~-1%,-1%~1%,1%-3%,>3%)
 In addition, we tried different sample period and pridiction period for the best performence and accuracy.
## Usage
Download the historical data and run "python auto_build_model.py" (it would take about a week for a single GTX-1070 Ti).
Download the recent data and run "python auto_test.py" (it would take about 3 hours for 1 stock market(NASDAQ or NYSE)).
