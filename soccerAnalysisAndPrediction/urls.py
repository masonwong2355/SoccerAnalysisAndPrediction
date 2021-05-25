from django.urls import path

from . import views

# app_name = 'soccerAnalysisAndPrediction'

urlpatterns = [
    path('', views.index, name='index'),
    path('eva/', views.eva, name='eva'),
    path('prediction/', views.prediction, name='prediction'),
    # path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
    path('predictResult/', views.predictResult, name='predictResult'),
    path('heatmap/', views.pol_MatchHeatmap, name='pol_MatchHeatmap'),
    path('passEvent/', views.plot_pass_events, name='plot_pass_events'),
    path('overviewStatistics/', views.overviewStatistics, name='overviewStatistics'),
    path('matchsCompetitions/<int:competition_id>/<int:season_id>', views.matchsCompetitions, name='matchsCompetitions'),
    path('matchAnalysis/<int:competition_id>/<int:season_id>/<int:match_id>', views.matchAnalysis, name='matchAnalysis'),
    # path('matchAnalysis/', views.matchAnalysis, name='matchAnalysis'),
    
    path('404/', views.error_404_view, name='error_404_view'),
    path('500/', views.error_500_view, name='error_500_view'),
]
