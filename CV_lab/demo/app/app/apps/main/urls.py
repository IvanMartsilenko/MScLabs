from django.urls import path
from django.conf.urls import url

from . import views
from django.contrib.auth.views import LogoutView
from django.conf import settings
from django.conf.urls.static import static

from django.conf.urls import include, url
from django.contrib import admin

app_name = 'main'

urlpatterns = [
    path('', views.main, name='main'),
    path('add_images', views.add_images, name='add_image'),
    path('display_images', views.display_images, name='display_images'),
    path('display_images/<int:img_id>/delete_image', views.delete_image, name='delete_image'),
path('download', views.download, name='download'),


    # path('success', views.success, name='success'),
    # path('signup', views.user_sign_up, name='user_sign_up'),
    # path('login', views.user_login, name='user_login'),
    # path("logout/", LogoutView.as_view(), name="logout"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
