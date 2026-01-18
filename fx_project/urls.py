"""
URL configuration for fx_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/X.Y/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include # <-- Make sure 'include' is imported

urlpatterns = [
    path('admin/', admin.site.urls), # Default Django admin site (we won't use it)
    
    # --- ADD THIS LINE ---
    # Any URL starting with 'forecast/' will be handled by the forecaster app's urls.py file
    path('forecast/', include('forecaster.urls')),
    # --- END ADD ---

    # Optional: If you want the root URL (e.g., http://127.0.0.1:8000/) 
    # to automatically go to the forecast page, you can add this later:
    # from django.views.generic import RedirectView
    # path('', RedirectView.as_view(url='/forecast/', permanent=True)),
]