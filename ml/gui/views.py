#### WEKA
import os, sys, getopt, re, random
import subprocess as sp
from datetime import datetime as dt
#### Django
from django.shortcuts import render
from django.views.generic import TemplateView
from django.http.response import JsonResponse
from gui.functions import send_to_gui
### THREADING
from threading import Thread

script_is_off = True

class Index(TemplateView):
    template_name = 'gui/index.html'

    def get_context_data(self):
        context = super().get_context_data()
        context['script_is_off'] = script_is_off
        return context


    def get(self, req, *args, **kwargs):
        return render(req, self.template_name, self.get_context_data())

    def post(self, req, *args, **kwargs):
        global script_is_off, weka, script_t, model

        if req.POST.get('command') == 'start' and script_is_off:
            try:
                script_is_off = False
            except:
                print('COULD NOT START SCRIPT')

        elif req.POST.get('command') == 'stop':
            script_is_off = True
            print('SCRIPT WILL STOP AFTER THIS RUN HAS ENDED')

            
        return JsonResponse({}, safe = False)