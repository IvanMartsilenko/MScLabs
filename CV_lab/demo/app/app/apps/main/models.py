from django.db import models


class Photo(models.Model):
    class Meta:
        verbose_name = 'Photo'
        verbose_name_plural = 'Photos'

    name = models.CharField(max_length=50, default='NONAME')
    image = models.ImageField(null=False, blank=False, upload_to='images/')
