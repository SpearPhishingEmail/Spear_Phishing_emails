from sqlalchemy import create_engine
import pymysql
import pandas as pd

sqlEngine       = create_engine('mysql+pymysql://root:root@127.0.0.1/machine_learning', pool_recycle=3600)

dbConnection    = sqlEngine.connect()

dataset           = pd.read_sql('CALL get_attentions_per_day();', dbConnection)

df = dataset.fillna(0)
dbConnection.close()