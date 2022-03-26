from django.conf.urls import url
from . import views
from django.views.generic import RedirectView
from django.views.generic import TemplateView
from django.urls import path, include,re_path

urlpatterns = [
    path('send-form-email/', views.SendFormEmail.as_view(), name='send_email'),
     url(r'^contact/$', TemplateView.as_view(template_name="contact.html"), name='contact'),
    url(r'^data/$', views.HomePageView.as_view(), name='home'),
     url(r'^faq/$', views.FaqPageView.as_view(), name='faq'),
    url(r'^info/(?P<fName>\w{0,50})/$',views.info, name='info'),
   url(r'^favicon\.ico$', RedirectView.as_view(url='/static/images/favicon.ico')),
    url(r'^prediction/(?P<fName>\w{0,50})/$', views.predictionCC, name='predictionCC'),
    url(r'^preprocess/(?P<fName>\w{0,50})/$',views.preprocess, name='preprocess'),
    url(r'^preprocess/(?P<fName>\w{0,50})/cleaning/$',views.cleaning, name='cleaning'),
    url(r'^preprocess/(?P<fName>\w{0,50})/normalization/$',views.normalization, name='normalization'),
    url(r'^preprocess/(?P<fName>\w{0,50})/label-encoding/$',views.labelEncoding, name='labelEncoding'),
    url(r'^visualizer3/(?P<fName>\w{0,50})/$',views.visualize1, name='visualize'),
    url(r'^download/processed/(?P<fName>\w{0,50})/$', views.downloadProcessed, name='downloadProcessed'),
    url(r'^view/(?P<fName>\w{0,50})/$',views.view, name='view'),
    #url(r'^cluster/(?P<fName>\w{0,50})/$',views.cluster, name='cluster'),
    #url(r'^classify/(?P<fName>\w{0,50})/$',views.classify, name='classify'),
    #url(r'^download/(?P<fName>\w{0,50})/$',views.downloadOriginal, name='downloadOriginal'),
    #url(r'^download/info/(?P<fName>\w{0,50})/$',views.downloadInfo, name='downloadInfo'),
    # url(r'^feature-selection/(?P<fName>\w{0,50})/$',views.featureSelection, name='featureSelection'),
    url(r'^$',views.upload, name='upload')
];