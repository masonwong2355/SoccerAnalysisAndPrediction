# Generated by Django 3.1.7 on 2021-02-28 08:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('soccerAnalysisAndPrediction', '0002_auto_20210228_1606'),
    ]

    operations = [
        migrations.AlterField(
            model_name='player',
            name='contract_valid_until',
            field=models.CharField(default=0, max_length=30),
        ),
    ]
