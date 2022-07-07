$(document).ready(function () {

    var airports = ["Agartala Airport", "Agatti Airport", "Agra Airport", "Akola Airport", "Allahabad Airport", "Along Airport", "Aurangabad Airport", "Bagdogra Airport", "Balurghat Airport", "Bareilly Air Force Station", "Basanth Nagar Airport", "Basanth Nagar Airport", "Begumpet Airport", "Belgaum Airport", "Bellary Airport", "Bathinda Airport", "Bhavnagar Airport", "Bhuj Airport", "Biju Patnaik Airport", "Bilaspur Airport", "Birsa Munda Airport", "Calicut International Airport", "Car Nicobar Air Force Station", "Chandigarh Airport", "Chaudhary Charan Singh International Airport", "Chennai International Airport", "Mumbai International Airport", "Cochin International Airport", "Coimbatore International Airport", "Cooch Behar Airport", "Dabolim Airport", "Daman Airport", "Daporijo Airport", "Dehradun Airport", "Devi Ahilyabai Holkar Airport", "Dhanbad Airport", "Dibrugarh Airport", "Dimapur Airport", "Diu Airport", "Dr. Babasaheb Ambedkar International Airport", "Gaya Airport", "Gorakhpur Airport", "Guna Airport", "Gwalior Airport", "Hisar Airport", "Hubli Airport", "Imphal Airport", "Indira Gandhi International Airport", "Jabalpur Airport", "Jaipur International Airport", "Jaisalmer Airport"]

    $("#Stop_1").hide();
    $("#Stop_2").hide();
    $("#Stop_3").hide();
    function getAirportsList(airports, StopId) {
        var mySelect = $(StopId);
        if ($(mySelect).has('option').length == 0) {
            $.each(airports, function (val, text) {
                mySelect.append(
                    $('<option></option>').val(val).html(text)
                );
            });
        }
    }


    $("#Total_Stops").change(function () {
        if ($("#Total_Stops").val() == "1 stop") {
            $("#Stop_1").show();
            $("#Stop_2").hide();
            $("#Stop_3").hide();
            getAirportsList(airports, '#Stop_1')

        }
        else if ($("#Total_Stops").val() == "2 stops") {
            $("#Stop_1").show();
            $("#Stop_2").show();
            $("#Stop_3").hide();
            getAirportsList(airports, '#Stop_1')
            getAirportsList(airports, '#Stop_2')
        }
        else if ($("#Total_Stops").val() == "3 stops") {
            $("#Stop_1").show();
            $("#Stop_2").show();
            $("#Stop_3").show();
            getAirportsList(airports, '#Stop_1')
            getAirportsList(airports, '#Stop_2')
            getAirportsList(airports, '#Stop_3')

        }
        else {
            $("#Stop_1").hide();
            $("#Stop_2").hide();
            $("#Stop_3").hide();
        }
    });



    $("#Source").change(function () {
        $('#Route_Source').val($("#Source").val());
    });
    $("#Destination").change(function () {
        $('#Route_Destination').val($('#Destination').val());
    });

});