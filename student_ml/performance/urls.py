from django.urls import path
from . import views  # This imports views.py from the same app folder

urlpatterns = [
    path('form/', views.performance_form, name='performance_form'),
    path('predict/', views.classify_and_predict, name='predict_performance'),
]
