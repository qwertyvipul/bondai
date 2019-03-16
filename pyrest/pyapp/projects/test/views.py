from django.http import HttpResponse
from . import hello

def req_hello(request):
    return HttpResponse(hello.printHello())