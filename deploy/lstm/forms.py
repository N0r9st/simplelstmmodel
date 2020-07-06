from django import forms

class ContactForm(forms.Form):
    #review = forms.CharField()
    review = forms.CharField(widget = forms.Textarea)