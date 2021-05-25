from .models import Player
from rest_framework import serializers


class PlayerSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Player
        fields = ['player_id', 'player_url', 'short_name', 'long_name']
