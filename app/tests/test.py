import pandas as pd

category_df = pd.read_csv("data/category_pipe.txt", sep="|", header=None, names=["catid", "catgroup", "catname", "catdesc"])
event_df = pd.read_csv("data/allevents_pipe.txt", sep="|", header=None, names=["eventid", "venueid", "catid","dateid",  "eventname", "starttime"])
venue_df = pd.read_csv("data/venue_pipe.txt", sep="|", header=None, names=["venueid", "venuename", "venuecity", "venuestate", "venueseats"])
listing_df = pd.read_csv("data/listings_pipe.txt", sep="|", header=None, names=["listid", "sellerid", "eventid", "dateid", "numtickets", "priceperticket", "totalprice", "listtime"])
date_df = pd.read_csv("data/date2008_pipe.txt", sep="|", header=None, names=["dateid", "caldate", "day", "week", "month", "qtr", "year", "holiday"])
users_df = pd.read_csv("data/allusers_pipe.txt", sep="|", header=None, names=["userid", "username", "firstname", "lastname", "city", "state", "email", "phone", "likesports", "liketheatre", "likeconcerts", "likejazz", "likeclassical", "likeopera", "likerock", "likevegas", "likebroadway", "likemusicals"])
sales_df = pd.read_csv("data/sales_tab.txt", sep="\t", header=None, names=["salesid", "listid", "sellerid", "buyerid", "eventid", "dateid", "qtysold", "pricepaid", "commission", "saletime"])

# Combinar las tablas
sales_event_df = pd.merge(sales_df, event_df, on="eventid")
sales_event_venue_df = pd.merge(sales_event_df, venue_df, on="venueid")
sales_event_venue_date_df = pd.merge(sales_event_venue_df, date_df, on="dateid", suffixes=("", "_date"))
sales_event_venue_date_category_df = pd.merge(sales_event_venue_date_df, category_df, on="catid")

# Procesar la columna 'saletime' para extraer la fecha sin la hora
sales_event_venue_date_category_df["sale_date"] = pd.to_datetime(sales_event_venue_date_category_df["saletime"]).dt.date

# Agrupar por 'sale_date' y sumar la cantidad vendida (qtysold)
daily_sales_df = sales_event_venue_date_category_df.groupby("sale_date").agg({"qtysold": "sum"}).reset_index()

# Ordenar por fecha
daily_sales_df.sort_values("sale_date", inplace=True)

# Extraiga las características (X) y la variable objetivo (y)
X = daily_sales_df["sale_date"].values.reshape(-1, 1)
y = daily_sales_df["qtysold"].values

# Divida los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)