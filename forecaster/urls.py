# forecaster/urls.py
from django.urls import path
from . import views

app_name = 'forecaster'

urlpatterns = [
    # /forecast/ maps to the LIVE view
    path('', views.forecast_view, name='forecast_page'),

    # /forecast/instant/ maps to the INSTANT view
    path('instant/', views.instant_forecast_view, name='instant_forecast_page'),
]