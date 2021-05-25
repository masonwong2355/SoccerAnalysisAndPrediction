from django.urls import path

from . import views

# app_name = 'csvs'

urlpatterns = [
    path('uploadPlayer/', views.upload_player_file_view, name='uploadPlayer'),
]
