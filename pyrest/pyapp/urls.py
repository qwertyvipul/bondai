from django.urls import include, path
from .projects.test import views as test_views

urlpatterns = [
    path('test/hello', test_views.req_hello)
]