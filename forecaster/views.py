from django.shortcuts import render
from django.contrib import messages 
import pandas as pd
from . import utils 
import time 

# View for LIVE TRAINING Forecast (Default Page) 
def forecast_view(request):
    """
    Handles the main forecast page display (/forecast/) and processes
    the LIVE TRAINING forecast request.
    """
    start_time = time.time() 

    # Initialize context dictionary
    context = {
        'currencies': utils.CURRENCIES,
        'forecast_data': None,
        'chart_html': None, # coz we are using plotly instead
        'from_currency': request.POST.get('from_currency', utils.CURRENCIES[utils.CURRENCIES.index("EUR")]),
        'to_currency': request.POST.get('to_currency', utils.CURRENCIES[utils.CURRENCIES.index("USD")]),

        'error_message': None,
        'execution_time': None,
    }

    # Handle POST request (form submission)
    if request.method == 'POST':
        from_currency = request.POST.get('from_currency')
        to_currency = request.POST.get('to_currency')

        # Update context with submitted values
        context['from_currency'] = from_currency
        context['to_currency'] = to_currency

        # Input Validation
        if not from_currency or not to_currency:
            messages.error(request, "Missing input values. Please select both currencies.")
            context['error_message'] = "Missing input values."
            return render(request, 'forecaster/forecast.html', context)

        if from_currency == to_currency:
            messages.error(request, "'From' and 'To' currencies cannot be the same.")
            context['error_message'] = "'From' and 'To' currencies cannot be the same."
            return render(request, 'forecaster/forecast.html', context)

        print(f"Processing LIVE forecast request: {from_currency} -> {to_currency}")

        # Forecasting Logic (LIVE TRAINING ONLY)
        final_forecast = None
        hist_df = None
        error_msg = None

        # Get Data for the specific pair requested
        hist_df, error_msg = utils.get_data(from_currency, to_currency)

        if hist_df is not None:
            # Build and Train the model LIVE
            # NOTE: This is the slow part! so be patient lol
            live_model, live_scaler, error_msg = utils.build_and_train_live(hist_df)

            # Perform forecast using the newly trained model if successful
            if live_model is not None:
                final_forecast, error_msg = utils.perform_rolling_forecast(live_model, live_scaler, hist_df)
        # If get_data failed, error_msg is already set

        # Process Results
        if final_forecast is not None and hist_df is not None:
            context['forecast_data'] = pd.DataFrame(final_forecast, columns=["Forecast"])
            hist_tail = hist_df['Rate'].tail(200) # Get recent history for plot context
            context['chart_html'] = utils.generate_plotly_html(hist_tail, final_forecast, from_currency, to_currency)
            if not context['chart_html']:
                error_msg = (error_msg + " | Failed to generate chart.") if error_msg else "Failed to generate chart."

        # If any step resulted in an error, display it
        if error_msg:
             messages.error(request, f"Live forecast failed: {error_msg}")
             context['error_message'] = f"Live forecast failed: {error_msg}"

        # Calculate execution time and add to context
        end_time = time.time()
        context['execution_time'] = round(end_time - start_time, 2)
        print(f"Live request processed in {context['execution_time']} seconds.")

        # Re-render the page with results (or errors) in the context
        return render(request, 'forecaster/forecast.html', context)

    # Handle GET request (initial page load)
    else:
        # Just display the initial page with the empty form
        print("Displaying initial LIVE forecast page (GET request).")
        return render(request, 'forecaster/forecast.html', context)


# View for INSTANT Forecast (Uses Pre-Trained Models)
def instant_forecast_view(request):
    # Handles the instant forecast page display (/forecast/instant/) and processes the forecast request using PRE-TRAINED models.
    start_time = time.time()
    context = {
        'currencies': utils.CURRENCIES,
        'forecast_data': None,
        'chart_html': None, 
        'from_currency': request.POST.get('from_currency', utils.CURRENCIES[utils.CURRENCIES.index("EUR")]),
        'to_currency': request.POST.get('to_currency', utils.CURRENCIES[utils.CURRENCIES.index("USD")]),
        'error_message': None,
        'execution_time': None,
        'is_instant_page': True 
    }

    if request.method == 'POST':
        from_currency = request.POST.get('from_currency')
        to_currency = request.POST.get('to_currency')
        context['from_currency'] = from_currency
        context['to_currency'] = to_currency

        # Input Validation
        if not from_currency or not to_currency:
            messages.error(request, "Please select both currencies.")
            context['error_message'] = "Please select both currencies."
            return render(request, 'forecaster/instant_forecast.html', context)

        if from_currency == to_currency:
            messages.error(request, "'From' and 'To' currencies cannot be the same.")
            context['error_message'] = "'From' and 'To' currencies cannot be the same."
            return render(request, 'forecaster/instant_forecast.html', context)

        print(f"Processing INSTANT forecast request: {from_currency} -> {to_currency}")

        # Forecasting Logic (PRE-TRAINED ONLY + Cross-Rate)
        final_forecast = None
        hist_df = None
        error_msg = None

        # cases
        # Case 1: Simple (e.g., EUR -> USD)
        if to_currency == utils.BASE_CURRENCY:
            model, scaler, error_msg = utils.load_forecasting_tools(from_currency, utils.BASE_CURRENCY)
            if not error_msg:
                hist_df, error_msg = utils.get_data(from_currency, utils.BASE_CURRENCY)
                if hist_df is not None:
                    final_forecast, error_msg = utils.perform_rolling_forecast(model, scaler, hist_df)

        # Case 2: Inverted (e.g., USD -> JPY)
        elif from_currency == utils.BASE_CURRENCY:
            # Load the OtherCurrency/USD model
            model, scaler, error_msg = utils.load_forecasting_tools(to_currency, utils.BASE_CURRENCY) 
            if not error_msg:
                # Get OtherCurrency/USD data
                hist_df_inverted, error_msg = utils.get_data(to_currency, utils.BASE_CURRENCY) 
                if hist_df_inverted is not None:
                    forecast_inverted, error_msg = utils.perform_rolling_forecast(model, scaler, hist_df_inverted)
                    # Calculate the final forecast and historical data if successful
                    if forecast_inverted is not None:
                        # Avoid division by zero
                        if not forecast_inverted.eq(0).any():
                            final_forecast = 1 / forecast_inverted
                            # Also calculate inverted historical for consistent plotting
                            if not hist_df_inverted['Rate'].eq(0).any():
                                hist_df = pd.DataFrame(1 / hist_df_inverted['Rate'], columns=['Rate'])
                            else:
                                error_msg = (error_msg + " | Cannot calculate historical rate (division by zero).") if error_msg else "Cannot calculate historical rate (division by zero)."
                                 # Ensure hist_df is None if calculation fails
                                hist_df = None
                        else:
                            error_msg = (error_msg + " | Forecast calculation failed (division by zero).") if error_msg else "Forecast calculation failed (division by zero)."

        # Case 3: Cross-Rate (e.g., GBP -> JPY)
        else:
            # Load models and data for FromCurrency/USD
            model_A, scaler_A, error_A = utils.load_forecasting_tools(from_currency, utils.BASE_CURRENCY)
            hist_A, error_hist_A = utils.get_data(from_currency, utils.BASE_CURRENCY)

            # Load models and data for ToCurrency/USD
            model_B, scaler_B, error_B = utils.load_forecasting_tools(to_currency, utils.BASE_CURRENCY)
            hist_B, error_hist_B = utils.get_data(to_currency, utils.BASE_CURRENCY)

            # Combine any errors encountered during loading/fetching
            error_msg = error_A or error_hist_A or error_B or error_hist_B

            # Proceed only if all necessary components loaded and fetched successfully
            if not error_msg:
                # Generate forecasts for both pairs
                forecast_A, error_fc_A = utils.perform_rolling_forecast(model_A, scaler_A, hist_A)
                forecast_B, error_fc_B = utils.perform_rolling_forecast(model_B, scaler_B, hist_B)
                # Combine forecast errors
                error_msg = error_msg or error_fc_A or error_fc_B 

                # Calculate cross-rate if both forecasts were successful
                if (forecast_A is not None) and (forecast_B is not None):
                    # Avoid division by zero
                    if not forecast_B.eq(0).any():
                        final_forecast = forecast_A / forecast_B
                        # Calculate historical cross-rate data, ensuring dates align
                        hist_A_aligned, hist_B_aligned = hist_A.align(hist_B, join='inner')
                        # Check division by zero for historical rates as well
                        if not hist_B_aligned['Rate'].eq(0).any():
                             hist_df = pd.DataFrame(hist_A_aligned['Rate'] / hist_B_aligned['Rate'], columns=['Rate'])
                        else:
                             error_msg = (error_msg + " | Historical cross-rate failed (division by zero).") if error_msg else "Historical cross-rate failed (division by zero)."
                             hist_df = None 
                    else:
                        error_msg = (error_msg + " | Forecast cross-rate failed (division by zero).") if error_msg else "Forecast cross-rate failed (division by zero)."

        # Process Results
        if final_forecast is not None and hist_df is not None:
            context['forecast_data'] = pd.DataFrame(final_forecast, columns=["Forecast"])
            hist_tail = hist_df['Rate'].tail(200)
            context['chart_html'] = utils.generate_plotly_html(hist_tail, final_forecast, from_currency, to_currency)
            if not context['chart_html']:
                error_msg = (error_msg + " | Failed to generate chart.") if error_msg else "Failed to generate chart."

        # If any step resulted in an error, display it using Django messages
        if error_msg:
             messages.error(request, f"Instant forecast failed: {error_msg}")
             context['error_message'] = f"Instant forecast failed: {error_msg}"

        # Calculate execution time
        end_time = time.time()
        context['execution_time'] = round(end_time - start_time, 2)
        print(f"Instant request processed in {context['execution_time']} seconds.")

        # Render the instant forecast template with results or errors
        return render(request, 'forecaster/instant_forecast.html', context)

    # Handle GET request (initial page load)
    else:
        # Just display the initial instant forecast page form
        print("Displaying initial INSTANT forecast page (GET request).")
        return render(request, 'forecaster/instant_forecast.html', context)