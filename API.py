import datetime

from flask import Flask, request, render_template
from openpyxl import load_workbook

from predict import makePrediction

app = Flask(__name__)


def getDurationaAndArrival(Date_Of_Departure, Date_of_Arrival, Dep_Time, Arrival_Time):
    Departure = datetime.datetime.strptime(Date_Of_Departure, '%Y-%m-%d')
    print(Departure)
    Dep_Time = datetime.datetime.strptime(Dep_Time, '%H:%M')
    print(Dep_Time.hour, Dep_Time.minute)
    Departure = datetime.datetime.combine(Departure.date(), datetime.time(Dep_Time.hour, Dep_Time.minute))
    print(Departure)
    Arrival = datetime.datetime.strptime(Date_of_Arrival, '%Y-%m-%d')
    Arrival_Time = datetime.datetime.strptime(Arrival_Time, '%H:%M')
    Arrival = datetime.datetime.combine(Arrival.date(), datetime.time(Arrival_Time.hour, Arrival_Time.minute))
    print(Arrival)

    Arrival = str(Arrival)
    Departure = str(Departure)

    datetimeFormat = '%Y-%m-%d %H:%M:%S'
    diff = datetime.datetime.strptime(Arrival, datetimeFormat) \
           - datetime.datetime.strptime(Departure, datetimeFormat)
    Arrival = datetime.datetime.strptime(Arrival, datetimeFormat)
    print("No. of month:", Arrival.month)
    month_abre = datetime.date(Arrival.year, Arrival.month, 1).strftime('%b')
    print(month_abre)
    print("Difference:", diff)
    days = diff.days
    print("Days:", days)

    hours, remainder = divmod(diff.seconds, 3600)
    hours = hours + (days * 24)
    minutes, seconds = divmod(remainder, 60)
    print("Hours: ", str(hours) + 'h')
    if minutes > 0:
        print("Minutes: ", str(minutes) + 'm')
    Arrival_Detail = str(Arrival_Time.hour) + ':' + str(Arrival_Time.minute)
    if days > 0:
        Arrival_Detail = Arrival_Detail + ' ' + str(Arrival.day) + ' ' + str(month_abre)
        print(Arrival_Detail)
    if minutes == 0:
        Duration = str(hours) + 'h'
    else:
        Duration = str(hours) + 'h' + ' ' + str(minutes) + 'm'
    return Duration, Arrival_Detail


def getRoute(Source, Destination, Stop_1, Stop_2, Stop_3, Stop_4):
    Route = ""
    airports = ["Netaji Subhash Chandra Bose International Airport", "Kempegowda International Airport Bengaluru",
                "Agartala Airport", "Agatti Airport", "Agra Airport", "Akola Airport", "Allahabad Airport",
                "Along Airport", "Aurangabad Airport", "Bagdogra Airport", "Balurghat Airport",
                "Bareilly Air Force Station", "Basanth Nagar Airport", "Basanth Nagar Airport", "Begumpet Airport",
                "Belgaum Airport", "Bellary Airport", "Bathinda Airport", "Bhavnagar Airport", "Bhuj Airport",
                "Biju Patnaik Airport", "Bilaspur Airport", "Birsa Munda Airport", "Calicut International Airport",
                "Car Nicobar Air Force Station", "Chandigarh Airport", "Chaudhary Charan Singh International Airport",
                "Chennai International Airport", "Mumbai International Airport", "Cochin International Airport",
                "Coimbatore International Airport", "Cooch Behar Airport", "Dabolim Airport", "Daman Airport",
                "Daporijo Airport", "Dehradun Airport", "Devi Ahilyabai Holkar Airport", "Dhanbad Airport",
                "Dibrugarh Airport", "Dimapur Airport", "Diu Airport", "Dr. Babasaheb Ambedkar International Airport",
                "Gaya Airport", "Gorakhpur Airport", "Guna Airport", "Gwalior Airport", "Hisar Airport",
                "Hubli Airport", "Imphal Airport", "Indira Gandhi International Airport", "Jabalpur Airport",
                "Jaipur International Airport", "Jaisalmer Airport"]
    Codes = ["CCU", "BLR", "IXA", "AGX", "AGR", "AKD", "IXD", "IXV", "IXU", "IXB", "RGH", "BEK", "RMD", "RMD", "BPM",
             "IXG", "BEP", "BUP", "BHU", "BHJ", "BBI", "PAB", "IXR", "CCJ", "CBD", "IXC", "LKO", "MAA", "BOM", "COK",
             "CJB", "COH", "GOI", "NMB", "DEP", "DED", "IDR", "DBD", "DIB", "DMU", "DIU", "NAG", "GAY", "GOP", "GUX",
             "GWL", "HSS", "HBX", "IMF", "DEL", "JLR", "JAI", "JSA"]
    Other = {"Delhi": "DEL", "New Delhi": "DEL", "Kolkata": "CCU", "Banglore": "BLR", "Mumbai": "BOM", "Chennai": "MAA",
             "Cochin": "COK",
             "Hyderabad": "BPM"}
    SourceAbr = Other[Source]
    DestinationAbr = Other[Destination]

    Stop_1Abr = None
    Stop_2Abr = None
    Stop_3Abr = None
    Stop_4Abr = None

    airportsCodes = {airports[i]: Codes[i] for i in range(len(airports))}  # Making airports and codes dictionary
    if Stop_1 in airportsCodes:
        Stop_1Abr = airportsCodes[Stop_1]
    if Stop_2 in airportsCodes:
        Stop_2Abr = airportsCodes[Stop_2]
    if Stop_3 in airportsCodes:
        Stop_3Abr = airportsCodes[Stop_3]
    if Stop_4 in airportsCodes:
        Stop_4Abr = airportsCodes[Stop_4]

    RouteList = [SourceAbr, Stop_1Abr, Stop_2Abr, Stop_3Abr, Stop_4Abr, DestinationAbr]
    for stop in RouteList:
        if stop != None:
            Route = Route + stop
            if RouteList[-1] != stop:
                Route = Route + " → "
    return Route


@app.route('/')
def index():
    return render_template('Insert.html')


@app.route('/Insert')
def Insert():
    return render_template('Insert.html')


@app.route('/Prediction')
def predict():
    return render_template('Predict.html')


@app.route('/Charts')
def Charts():
    return render_template('Charts.html')


@app.route("/insert", methods=["POST"])
def submit():
    if request.method == "POST":
        userdata = dict(request.form)
        Airline = userdata["Airline"]
        Date_of_Departure = userdata["Date_of_Departure"]
        Total_Stops = userdata["Total_Stops"]
        Additional_Info = userdata["Additional_Info"]
        Price = int(userdata["Price"])
        Dep_Time = userdata["Dep_Time"]
        Arrival_Time = userdata["Arrival_Time"]
        Source = userdata["Source"]
        Destination = userdata["Destination"]
        Date_of_Arrival = userdata["Date_of_Arrival"]
        Stop_1 = None
        Stop_2 = None
        Stop_3 = None
        Stop_4 = None
        Duration, Arrival_Time = getDurationaAndArrival(Date_of_Departure, Date_of_Arrival, Dep_Time, Arrival_Time)
        Date_of_Departure = datetime.datetime.strptime(Date_of_Departure, '%Y-%m-%d')
        Date_of_Departure = '{0}/{1:02}/'.format(Date_of_Departure.month, Date_of_Departure.day % 100) + str(
            Date_of_Departure.year)
        print(Date_of_Departure)
        try:
            Stop_1 = userdata["Stop_1"]
            Stop_2 = userdata["Stop_2"]
            Stop_3 = userdata["Stop_3"]
            Stop_4 = userdata["Stop_4"]
        except:
            print("Less than 4 Stops have been provided..")
        print("Stops:", Stop_1, Stop_2, Stop_3, Stop_4)
        print(Total_Stops)
        Route = getRoute(Source, Destination, Stop_1, Stop_2, Stop_3, Stop_4)
        print(Route)

        workbook_name = 'Data_Train.xlsx'
        wb = load_workbook(workbook_name)
        page = wb.active

        # New data to write:
        info = [Airline, Date_of_Departure, Source, Destination, Route, Dep_Time, Arrival_Time, Duration,
                Total_Stops, Additional_Info, Price]

        page.append(info)

        wb.save(filename=workbook_name)
    info = "Data Written Successfully!\n" + str(info)
    return render_template("Insert.html", content=info)


@app.route("/predict", methods=["POST"])
def submitPredict():
    if request.method == "POST":
        userdata = dict(request.form)
        Airline = userdata["Airline"]
        Date_of_Departure = userdata["Date_of_Departure"]
        Total_Stops = userdata["Total_Stops"]
        Additional_Info = userdata["Additional_Info"]
        Dep_Time = userdata["Dep_Time"]
        Arrival_Time = userdata["Arrival_Time"]
        Source = userdata["Source"]
        Destination = userdata["Destination"]
        Date_of_Arrival = userdata["Date_of_Arrival"]
        Stop_1 = None
        Stop_2 = None
        Stop_3 = None
        Stop_4 = None
        Duration, Arrival_Time = getDurationaAndArrival(Date_of_Departure, Date_of_Arrival, Dep_Time, Arrival_Time)
        Date_of_Departure = datetime.datetime.strptime(Date_of_Departure, '%Y-%m-%d')
        Date_of_Departure = '{0}/{1:02}/'.format(Date_of_Departure.month, Date_of_Departure.day % 100) + str(
            Date_of_Departure.year)
        print(Date_of_Departure)
        try:
            Stop_1 = userdata["Stop_1"]
            Stop_2 = userdata["Stop_2"]
            Stop_3 = userdata["Stop_3"]
            Stop_4 = userdata["Stop_4"]
        except:
            print("Less than 4 Stops have been provided..")
        print("Stops:", Stop_1, Stop_2, Stop_3, Stop_4)
        print(Total_Stops)
        Route = getRoute(Source, Destination, Stop_1, Stop_2, Stop_3, Stop_4)
        print(Route)
    prediction = makePrediction(df={'Airline': Airline, 'Date_of_Journey': Date_of_Departure, 'Source': Source,
                                    'Destination': Destination, 'Route': Route, 'Dep_Time': Dep_Time,
                                    'Arrival_Time': Arrival_Time, 'Duration': Duration,
                                    'Total_Stops': Total_Stops, 'Additional_Info': Additional_Info})
    prediction = "Predicted Price: Rs."+ prediction

    return render_template("Predict.html", content=prediction)


if __name__ == '__main__':
    app.run()
