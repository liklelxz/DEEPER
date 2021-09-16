python model/arima_response.py --data_path ./Data/With_Features/fire.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/With_Features/law.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/With_Features/structural.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/With_Features/utility.csv --percentage 0.5 --validation 1

python model/arima_response.py --data_path ./Data/DropCloseDate_Quant/fire.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/DropCloseDate_Quant/law.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/DropCloseDate_Quant/structural.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/DropCloseDate_Quant/utility.csv --percentage 0.5 --validation 1

python model/arima_response.py --data_path ./Data/Median_Quant/fire.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/Median_Quant/law.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/Median_Quant/structural.csv --percentage 0.5 --validation 1
python model/arima_response.py --data_path ./Data/Median_Quant/utility.csv --percentage 0.5 --validation 1
