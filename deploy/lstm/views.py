from django.shortcuts import render

from django.http import HttpResponse, JsonResponse
from .apps import *
from .forms import ContactForm
from .model_functions import predict, model

def call_model(request):
    print('VIEWSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
    response = 'Please, write your movie review.'
    sentiment = 'Note that the estimation might be more precise if the review has more than one sentence.'
    
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            review = form.cleaned_data['review']
            
            response = predict(model, review)[-1].detach().numpy()
            
            if response >= .5:
                response = round(7 + 6 * (response - 0.5))
                sentiment = 'Positive review'
            else:
                response = round(4 - 6 * (-response + 0.5))
                sentiment = 'Negative review'
            
        
    form = ContactForm()
    
    
    return render(request, 'form.html', {'form':form, 'response':response, 'sentiment' : sentiment})