from django.urls import path
from gui.views import Index

urlpatterns = [
    path('', Index.as_view(), name = "Index"),

]
