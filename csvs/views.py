from django.shortcuts import render
from .forms import CsvModelForm
from .models import CSV
import csv
from django.contrib.auth.models import User
from soccerAnalysisAndPrediction.models import Player
# from django.http import HttpResponse

# Create your views here.
# @csrf_protect
def upload_player_file_view(request):
    form = CsvModelForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()
        form = CsvModelForm()
        obj = CSV.objects.get(activated=False)
        with open(obj.file_name.path, 'r',encoding="utf-8") as f:
            reader = csv.reader(f)

            obj.activated = True
            obj.save()
            for i, row in enumerate(reader):
                if i == 0:
                    pass
                else:
                    # sumPare = [ls, st, rs, lw, lf, cf, rf, rw, la, ca, ra, lm, lc, cm, rc, rm, lw, ld, cd, rd, rw, lb, lc, cb, rc, rb]
                    for num in range(80, 106):
                        scroe = row[num].split("+")
                        scroe = int(scroe[0]) + int(scroe[1]) if scroe[1] != '' else int(scroe[0])
                        row[num] = scroe
                                        
                    # Entry.objects.filter(pub_date__year=2006)
                    Player.objects.create(
                        player_id                  = row[0],
                        player_url                 = row[1],
                        short_name                 = row[2],
                        long_name                  = row[3],
                        age                        = row[4],
                        dob                        = row[5],
                        height_cm                  = row[6],
                        weight_kg                  = row[7],
                        nationality                = row[8],
                        club_name                  = row[9],
                        league_name                = row[10],
                        league_rank                = row[11],
                        overall                    = row[12],
                        potential                  = row[13],
                        value_eur                  = row[14],
                        wage_eur                   = row[15],
                        player_positions           = row[16],
                        preferred_foot             = row[17],
                        international_reputation   = row[18],
                        weak_foot                  = row[19],
                        skill_moves                = row[20],
                        work_rate                  = row[21],
                        body_type                  = row[22],
                        real_face                  = row[23],
                        release_clause_eur         = row[24] == 0 if row[24] == '' else row[24],
                        player_tags                = row[25],
                        team_position              = row[26],
                        team_jersey_number         = row[27],
                        loaned_from                = row[28],
                        joined                     = row[29],
                        contract_valid_until       = row[30] == 0 if row[30] == '' else row[30],
                        nation_position            = row[31],
                        nation_jersey_number       = row[32] == 0 if row[32] == '' else row[32],
                        pace                       = row[33] == 0 if row[33] == '' else row[33],
                        shooting                   = row[34] == 0 if row[34] == '' else row[34],
                        passing                    = row[35] == 0 if row[35] == '' else row[35],
                        dribbling                  = row[36] == 0 if row[36] == '' else row[36],
                        defending                  = row[37] == 0 if row[37] == '' else row[37],
                        physic                     = row[38] == 0 if row[38] == '' else row[38],
                        gk_diving                  = row[39] == 0 if row[39] == '' else row[39],
                        gk_handling                = row[40] == 0 if row[40] == '' else row[40],
                        gk_kicking                 = row[41] == 0 if row[41] == '' else row[41],
                        gk_reflexes                = row[42] == 0 if row[42] == '' else row[42],
                        gk_speed                   = row[43] == 0 if row[43] == '' else row[43],
                        gk_positioning             = row[44] == 0 if row[44] == '' else row[44],
                        player_traits              = row[45],
                        attacking_crossing         = row[46],
                        attacking_finishing        = row[47],
                        attacking_heading_accuracy = row[48],
                        attacking_short_passing    = row[49],
                        attacking_volleys          = row[50],
                        skill_dribbling            = row[51],
                        skill_curve                = row[52],
                        skill_fk_accuracy          = row[53],
                        skill_long_passing         = row[54],
                        skill_ball_control         = row[55],
                        movement_acceleration      = row[56],
                        movement_sprint_speed      = row[57],
                        movement_agility           = row[58],
                        movement_reactions         = row[59],
                        movement_balance           = row[60],
                        power_shot_power           = row[61],
                        power_jumping              = row[62],
                        power_stamina              = row[63],
                        power_strength             = row[64],
                        power_long_shots           = row[65],
                        mentality_aggression       = row[66],
                        mentality_interceptions    = row[67],
                        mentality_positioning      = row[68],
                        mentality_vision           = row[69],
                        mentality_penalties        = row[70],
                        mentality_composure        = row[71],
                        defending_marking          = row[72],
                        defending_standing_tackle  = row[73],
                        defending_sliding_tackle   = row[74],
                        goalkeeping_diving         = row[75],
                        goalkeeping_handling       = row[76],
                        goalkeeping_kicking        = row[77],
                        goalkeeping_positioning    = row[78],
                        goalkeeping_reflexes       = row[79],
                        ls                         = row[80],
                        st                         = row[81],
                        rs                         = row[82],
                        lw                         = row[83],
                        lf                         = row[84],
                        cf                         = row[85],
                        rf                         = row[86],
                        rw                         = row[87],
                        lam                        = row[88],
                        cam                        = row[89],
                        ram                        = row[90],
                        lm                         = row[91],
                        lcm                        = row[92],
                        cm                         = row[93],
                        rcm                        = row[94],
                        rm                         = row[95],
                        lwb                        = row[96],
                        ldm                        = row[97],
                        cdm                        = row[98],
                        rdm                        = row[99],
                        rwb                        = row[100],
                        lb                         = row[101],
                        lcb                        = row[102],
                        cb                         = row[103],
                        rcb                        = row[104],
                        rb                         = row[105]
                    )

                    # print(row)
                print('done')
            
    return render(request, 'csvs/upload.html', {'form': form})