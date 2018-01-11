import numpy as np
import matplotlib.pyplot as plt
import math as math
import scipy.interpolate as interpld
from pyremind import *
from pygcmtreat import *
from pykcorr import *
from pyfunction import *
from pydataread import *
from pyconstant import *
from netCDF4 import *
from netCDF4 import Dataset
import math as math
import os,sys
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
number_rank = comm.size

########################################################################################################################
########################################################################################################################

"""
    PYTRANSFERT

    Cette bibliotheque intervient a deux niveaux : premierement lors dela resolution du transfert radiatif en
    transmission, puis dans la generation des spectres d'absorption ou des courbes de lumiere.

    Les fonctions ci-dessous appellent des fichiers de donnee precalcules notamment par les routines de pkcorr. Elles
    exploitent egalement l'optimisation apportee par pyremind qui allege considerablement le nombre de calculs et par
    consequent, les temps de calcul.

    La deuxieme partie des fonctions ici definies servent a interpreter les cartes de transmittances produites par les
    fonction trans2fert. Seule la transmission est ici prise en compte, un autre module aura pour responsabilite d'ajouter
    l'emission a ces cartes

    Version : 6.0

    Recentes mises a jour :

    >> Suppression du retour de N dans altitude_line_array1D_cyl_optimized_correspondance
    >> Suppression de l'entree T dans altitude_line_array1D_cyl_optimized_correspondance
    >> Modification d'atmospectre, desormais si data_convert a ete calcule des le depart sur la base des milieux de
    couche le spectre est calcule sur cette base, les aires associees sont donc plus realistes.
    >> Correction sur les equations d'effective_radius qui ne calculait pas correctement les aires de recouvrement
    >> Suppression de M_air et g dans les parametres d'entree des routines trans2fert1D et trans2fert3D

    Date de derniere modification : 10.10.2016

    >> Refonte des fonctions de transfert radiatif et introduction de la possibilite d'integrer la profondeur optique
    >> sur le chemin optique des rayons. Prise en compte des differentes interpolations possibles par rapport a la
    >> temperature dans convertator1D
    >> Possibilite dans le cas discret de moduler la densite de reference grace a la fonction Module
    >> Fini pour les effets de bord et les effets aux poles
    >> Verification d'atmospectre et possibilite de soumettre des grilles de transmittance sur un point theta ou sur une
    >> bande unique

    Date de derniere modification : 12.12.2016

"""

########################################################################################################################
########################################################################################################################

"""
    TRANS2FERT1D

    Cette fonction exploite l'ensemble des outils developpes precedemment afin de produire une carte de transmittance
    dans une maille cylindrique. Si la maille n'est necessaire en soit, une seule longitude est utile dans la production
    des transits ou dans les estimations du rayon effectif (au premier ordre). Cette routine peut effectuer une
    interpolation sur les donnees, toutefois le temps de calcul est tres nettement augmente (plusieurs dizaines d'heure)
    L'utilisation des donnees brutes peut etre traite en utilisant ou non les tables dx_grid et order_grid pre-etablie.
    En fonction de la resolution initiale adoptee pour les donnees GCM, les tables dx et order permettent un petit gain
    de temps (pour les resolutions elevees).

    La production de la colonne peut etre effectuee en amont eventuellement.

    Cette fonction retourne la grille de transmittance dans une maille cylindrique I[bande,r,theta].

"""

########################################################################################################################
########################################################################################################################


def trans2fert1D (k_corr_data_grid,k_cont_h2h2,k_cont_h2he,k_cont_nu,T_cont,Q_cloud,Rp,h,g0,r_step,theta_step,\
                  x_step,gauss,gauss_val,dim_bande,data,P_col,T_col,gen_col,Q_col,compo_col,ind_active,dx_grid,order_grid,pdx_grid,\
                  P_sample,T_sample,Q_sample,bande_sample,name_file,n_species,c_species,single,\
                  bande_cloud,r_eff,r_cloud,rho_p,t,phi_rot,domain,ratio,lim_alt,rupt_alt,directory,z_grid,type,\
                  Tracer=False,Continuum=True,Molecular=False,Scattering=True,Clouds=False,Kcorr=True,Rupt=False,\
                  Middle=False,Integral=False,Module=False,Optimal=False,D3Maille=False) :

    r_size,theta_size,x_size = np.shape(dx_grid)
    number_size,z_size,lat_size,long_size = np.shape(data)

    Composition = True

    if Kcorr == True :

        k_rmd = np.zeros((T_col.size,dim_bande,gauss.size))

    else :

        k_rmd = np.zeros((T_col.size,dim_bande))

    Itot = np.ones((dim_bande,r_size,1))

    bar = ProgressBar(r_size,'Radiative transfert progression')

    for i in range(1) :

        if i == 0 :

            theta_line = i

            if Rupt == True :

                dep = int(rupt_alt/r_step)

                Itot[:, 0:dep, theta_line] = np.zeros((dim_bande,dep))

            else :

                dep = 0

            for j in range(dep,r_size) :

                if Middle == False :

                    r = Rp + j*r_step

                else :

                    r = Rp + (j+0.5)*r_step
                r_line = j

                dx = dx_grid[r_line,theta_line,:]
                order = order_grid[:,r_line,theta_line,:]
                pdx = pdx_grid[r_line,theta_line,:]

                if r < Rp + lim_alt :

                    if j == 0 :

                        P_rmd,T_rmd,Q_rmd,k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd = \
                        convertator1D (P_col,T_col,gen_col,c_species,Q_col,compo_col,ind_active,\
                        k_corr_data_grid,k_cont_h2h2,k_cont_h2he,k_cont_nu,Q_cloud,T_cont,P_sample,\
                        T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name_file,t,phi_rot,\
                        n_species,domain,ratio,directory,Tracer,Composition,Continuum,Scattering,Clouds,Kcorr,Optimal)

                    zone, = np.where(dx >= 0)
                    cut, = np.where(order[0,zone] < z_size)
                    dx_ref = dx[zone[cut]]
                    #data_ref = data[:,order[0,zone[cut]],order[1,zone[cut]],order[2,zone[cut]]]
                    data_ref = data[:,order[0,zone[cut]],0,0]
                    P_ref, T_ref = data_ref[0], data_ref[1]

                    if Tracer == True :

                        Q_ref = data_ref[2]

                        k_inter,k_cont_inter,k_sca_inter,k_cloud_inter = \
                        k_correlated_interp_remind_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,\
                        P_ref.size,P_rmd,P_ref,T_rmd,T_ref,Q_rmd,Q_ref,Continuum,Scattering,Clouds,Kcorr)

                    else :

                        k_inter,k_cont_inter,k_sca_inter,k_cloud_inter = \
                        k_correlated_interp_remind(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,\
                        P_ref.size,P_rmd,P_ref,T_rmd,T_ref,Continuum,Scattering,Clouds,Kcorr)

                    if Module == True :
                        z_ref = z_grid[r_line,theta_line,order[3,zone[cut]]]
                        P_ref = module_density(P_ref,T_ref,z_ref,Rp,g0,data_ref[number_size-1],r_step,type,Middle)
                    Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

                    I_out = radiative_transfert_remind3D(dx_ref,pdx[zone[cut]],Cn_mol_ref,k_inter,k_cont_inter,k_sca_inter,\
                            k_cloud_inter,gauss_val,single,Continuum,Molecular,Scattering,Clouds,Kcorr,Integral)

                    Itot[:, r_line, 0] = I_out[:]

                bar.animate(j+1)

    return Itot


########################################################################################################################
########################################################################################################################

"""
    TRANS2FERT3D

    Cette fonction exploite l'ensemble des outils developpes precedemment afin de produire une carte de transmittance
    dans une maille cylindrique. Cette routine peut effectuer une interpolation sur les donnees, toutefois le temps de
    calcul est tres nettement augmente (plusieurs dizaines d'heure)
    L'utilisation des donnees brutes peut etre traite en utilisant ou non les tables dx_grid et order_grid pre-etablie.
    En fonction de la resolution initiale adoptee pour les donnees GCM, les tables dx et order permettent un petit gain
    de temps (pour les resolutions elevees).

    La production de la colonne peut etre effectuee en amont eventuellement.

    Cette fonction retourne la grille de transmittance dans une maille cylindrique I[bande,r,theta].

"""

########################################################################################################################
########################################################################################################################


def trans2fert3D (k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,Rp,h,g0,r_step,theta_step,gauss_val,dim_bande,data,\
                  P_rmd,T_rmd,Q_rmd,dx_grid,order_grid,pdx_grid,z_grid,t,\
                  name_file,n_species,single,rmind,lim_alt,rupt_alt,rank,rank_ref,\
                  Tracer=False,Continuum=True,Molecular=False,Scattering=True,Clouds=True,Kcorr=True,\
                  Rupt=False,Module=False,Integral=False,TimeSel=False) :

    r_size,theta_size,x_size = np.shape(dx_grid)
    number_size,t_size,z_size,lat_size,long_size = np.shape(data)

    if TimeSel == True :
        data = data[:,t,:,:,:]

    Itot = np.ones((dim_bande,r_size,theta_size))

    if rank == rank_ref : 
        bar = ProgressBar(theta_size,'Radiative transfert progression')

    for i in range(theta_size) :

        theta_line = i
        fail = 0

        if Rupt == True :

            dep = int(rupt_alt/r_step)

            Itot[:, 0:dep, theta_line] = np.zeros((dim_bande,dep))

        else :

            dep = 0

        for j in range(dep,r_size) :

            r = Rp + j*r_step
            r_line = j

            dx = dx_grid[r_line,theta_line,:]
            order = order_grid[:,r_line,theta_line,:]
            if Integral == True :
                pdx = pdx_grid[r_line,theta_line,:]

            if r < Rp + lim_alt :

                zone, = np.where((order[0] > 0)*(order[0] < z_size))
                dx_ref = dx[zone]
                if Integral == True :
                    pdx_ref = pdx[zone]
                else :
                    pdx_ref = np.array([])
                data_ref = data[:,order[0,zone],order[1,zone],order[2,zone]]
                P_ref, T_ref = data_ref[0], data_ref[1]

                if Tracer == True :

                    Q_ref = data_ref[2]

                    k_inter,k_cont_inter,k_sca_inter,k_cloud_inter,fail = \
                    k_correlated_interp_remind3D_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,P_ref.size,\
                    P_rmd,P_ref,T_rmd,T_ref,Q_rmd,Q_ref,n_species,fail,rmind,Continuum,Molecular,Scattering,Clouds,Kcorr)

                else :

                    k_inter,k_cont_inter,k_sca_inter,k_cloud_inter,fail = \
                    k_correlated_interp_remind3D(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,P_ref.size,\
                    P_rmd,P_ref,T_rmd,T_ref,n_species,fail,rmind,Continuum,Molecular,Scattering,Clouds,Kcorr)

                if Module == True :
                    z_ref = z_grid[r_line,theta_line,order[3,zone]]
                    P_ref = module_density(P_ref,T_ref,z_ref,Rp,g0,data_ref[number_size-1],r_step,type,True)
                Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

                I_out = radiative_transfert_remind3D(dx_ref,pdx_ref,Cn_mol_ref,k_inter,k_cont_inter,k_sca_inter,\
                    k_cloud_inter,gauss_val,single,Continuum,Molecular,Scattering,Clouds,Kcorr,Integral)

                Itot[:, r_line, theta_line] = I_out

        if rank == rank_ref : 
            bar.animate(i+1)

        if fail !=0 and rank == rank_ref :

            print("%i failure(s) at this latitude" %(fail))

    return Itot


########################################################################################################################
########################################################################################################################

"""
    EFFECTIVE_RADIUS

    Dans le cas d'une etoile uniforme en luminosite et parfaitement spherique, nous cherchons a determiner le rayon
    effectif d'une exoplanete transitant devant l'etoile. Nous integrons alors la grille cylindrique de transmittance
    et nous la soustrayons a la luminosite totale normalisee de l'etoile. Nous considerons pour cela une intensite de
    1 par metre carre.

    Cette fonction retourne le rayon effectif ainsi que l'epaisseur de l'atmosphere afin de verifier la pertinence du
    resultat.

"""

########################################################################################################################
########################################################################################################################


def effective_radius(I,R_s,Rp,r_step,extra,Middle=False) :

    I_p = 0.
    I_p2 = 0.
    r_size,theta_size = np.shape(I)
    A_surf = np.pi*Rp**2
    A_surf_arr = np.zeros((r_size,theta_size),dtype='float')
    I_sol = np.pi*R_s**2
    R = Rp

    for r in range(r_size) :

        if r == 0 :

            if Middle == False :
                A = np.pi*(1/2.)*(2.*R + 1/2.*r_step)*r_step/float(theta_size)
                R += r_step/2.
            else :
                A = 2.*np.pi*r_step*((r+0.5)*r_step + Rp)

            A_surf += A
            A_surf_arr[r,:] = np.ones(theta_size,dtype='float')*A/np.float(theta_size)

        else :

            if Middle == False :
                A = np.pi*(2*R + r_step)*r_step/float(theta_size)
                R += r_step
            else :
                A = 2.*np.pi*r_step*((r+0.5)*r_step + Rp)

            A_surf += A
            A_surf_arr[r,:] = np.ones(theta_size,dtype='float')*A/np.float(theta_size)

        for theta in range(theta_size) :

            I_p += I[r,theta]*A/np.float(theta_size)
            I_p2 += (1-I[r,theta])*A/np.float(theta_size)

    I_tot = I_sol - A_surf + I_p

    R_p = R_s*np.sqrt((I_sol - I_tot)/(I_sol))

    h_eff = R_p - Rp

    return R_p, h_eff, A_surf, A_surf_arr, I_p2


########################################################################################################################


def effective_radius_boucle(I,R_s,Rp,r_step,A_surf,A_surf_arr,extra) :

    I_sol = np.pi*R_s**2

    I_p = np.nansum(I*A_surf_arr)

    I_tot = I_sol - A_surf + I_p

    h_eff = - Rp +np.sqrt(A_surf/(np.pi)) - extra*r_step

    R_p = R_s*np.sqrt((I_sol - I_tot)/(I_sol))

    return R_p, h_eff


########################################################################################################################


def atmospectre(I,bande_sample,R_s,Rp,r_step,extra,trans,Kcorr=False,Middle=False) :

    if Kcorr == True :

        bande = np.array([])
        bande_bar = np.array([])

        dim_bande = bande_sample.size - 1

        for i in range(dim_bande+1) :

            if i == 0 :

                bande = np.append(bande, np.array([bande_sample[i]]))
                bande_bar = np.append(bande_bar, np.array([bande_sample[i]]))

            else :

                bande = np.append(bande, np.array([bande_sample[i]]))
                bande_bar = np.append(bande_bar, np.array([bande_sample[i]]))
                bande_bar = np.append(bande_bar, np.array([bande_sample[i]]))

        bande_bar = np.delete(bande_bar,bande_bar.size-1)

        R_eff = np.array([])
        R_eff_bar = np.array([])
        ratio = np.array([])
        ratio_bar = np.array([])

        for i in range(dim_bande) :

            if i == 0 :

                if trans == False :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(I[i,:,:],R_s,Rp,r_step,extra,Middle)
                else :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(np.transpose(I[i,:,:]),R_s,Rp,r_step,extra,Middle)

            else :

                if trans == False :
                    R,h = effective_radius_boucle(I[i,:,:],R_s,Rp,r_step,A_surf,A_surf_arr,extra)
                else :
                    R,h = effective_radius_boucle(np.transpose(I[i,:,:]),R_s,Rp,r_step,A_surf,A_surf_arr,extra)

            R_eff = np.append(R_eff, np.array([R]))
            R_eff_bar = np.append(R_eff_bar, np.array([R]))
            R_eff_bar = np.append(R_eff_bar, np.array([R]))

        R_ref = np.amin(R_eff)

        ratio_bar = np.append(ratio_bar, (R_eff_bar - R_ref)/R_ref*1000000.)
        ratR_bar = (R_eff_bar - R_ref)/R_s*1000000.

        flux_bar = R_eff_bar**2/R_s**2
        flux = 0

    else :

        dim_bande = bande_sample.size

        R_eff = np.array([])

        for i in range(dim_bande) :

            if i == 0 :

                if trans == False :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(I[i,:,:],R_s,Rp,r_step,extra,Middle)
                else :
                    R,h,A_surf,A_surf_arr,Ip = effective_radius(np.transpose(I[i,:,:]),R_s,Rp,r_step,extra,Middle)

            else :

                if trans == False :
                    R,h = effective_radius_boucle(I[i,:,:],R_s,Rp,r_step,A_surf,A_surf_arr,extra)
                else :
                    R,h = effective_radius_boucle(np.transpose(I[i,:,:]),R_s,Rp,r_step,A_surf,A_surf_arr,extra)

            R_eff = np.append(R_eff, np.array([R]))

        R_eff_bar = 0
        ratio_bar = 0
        ratR_bar = 0
        bande_bar = 0

        flux = R_eff**2/R_s**2
        flux_bar = 0

    return R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux


########################################################################################################################


def stellarint(Rs,U,param,Law) :

    Is = 0

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0]*(1 - np.cos(q)) - U[1]*(1 - np.cos(q))**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0]*(1 - np.sqrt(q)) - U[1]*(1 - q) - U[2]*(1 - q**(3/2.)) - U[3]*(1 - q**2)

    for i in range (int(Rs/param)) :
        d = i*param/Rs
        Is += I2(d,U)*np.pi*((d*Rs+param)**2 - (d*Rs)**2)

    return Is


########################################################################################################################


def stellarint_matrix (Rs,U,param,Law) :

    dim_coeff,dim_ban = np.shape(U)
    Is = np.zeros(dim_ban)

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0,:]*(1 - np.cos(q)) - U[1,:]*(1 - np.cos(q))**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0,:]*(1 - np.sqrt(q)) - U[1,:]*(1 - q) - U[2,:]*(1 - q**(3/2.)) - U[3,:]*(1 - q**2)

    for i in range (int(Rs/param)) :
        d = i*param/Rs
        Is += I2(d,U)*np.pi*((d*Rs+param)**2 - (d*Rs)**2)

    return Is

########################################################################################################################


def stellarplot (Rs,U,param,Law) :

    width = int(Rs/param)
    Is = np.zeros(width)

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0]*(1 - np.cos(q)) - U[1]*(1 - np.cos(q))**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0]*(1 - np.sqrt(np.cos(q))) - U[1]*(1 - np.cos(q)) - U[2]*(1 - np.cos(q)**(3/2.)) - U[3]*(1 - np.cos(q)**2)

    for i in range (int(Rs/param)) :
        d = i*param/Rs
        Is[i] = I2(d,U)

    Is_png = np.zeros((2*width+1,2*width+1))

    for i in range(width+1) :

        for j in range(width+1) :

            rho = np.sqrt(i**2 + j**2)
            index = int(np.round(rho))

            if index < width :

                Is_png[width+i,width+j] = Is[index]
                Is_png[width-i,width+j] = Is[index]
                Is_png[width+i,width-j] = Is[index]
                Is_png[width-i,width-j] = Is[index]

        if i%10 == 0 :

            print(i)

    print(Is)

    plt.imshow(Is_png,aspect='auto',extent=[-width*param,width*param,width*param,-width*param])
    plt.colorbar()
    plt.show()

    return Is, Is_png



########################################################################################################################


def lightcurve_maker(Rs,Rp,h,Ms,cDcorr,cPcorr,inc_0,U_limb,Itot_plan,r_step,theta_step,reso_occ,t,i_sample,bande_sample,bande_limb,Law) :

    reso_bande,reso_r,reso_theta = np.shape(Itot_plan)

    I_time = np.ones((i_sample.size,t.size))
    I_ground_time = np.zeros((i_sample.size,t.size))
    I_ground_plan_time = np.zeros((i_sample.size,t.size))
    I_star_time = np.ones((i_sample.size,t.size))

    a = ((((cPcorr*24*3600)**2)*G*(Ms/(4*np.pi**2)))**(1/3.))/(Rs)
    r = (Rp + h)/Rs

    ksi = 2*np.pi*(t - cDcorr)/cPcorr
    u = - np.cos(inc_0)*np.cos(ksi)
    v = np.sin(ksi)
    sep = a*np.sqrt(u**2 + v**2)


    #B2 = lambda q,ld1,ld2,sep,r : 2*q*I2(q,ld1,ld2)*np.arccos((q**2 + sep**2 - r**2)/(2*sep*q))
    #Fs2 = np.pi*(1 - ld1/3. - ld2/2.)

    for i_bande in range(i_sample.size) :

        Itot = Itot_plan[i_sample[i_bande],:,:]
        wave = 1./(bande_sample[i_sample[i_bande]]*100)*10**6
        wh, = np.where(bande_limb >= wave)

        if wh.size != 0 :

            index = wh[0] - 1

        else :

            index = bande_limb.size - 2

        U = U_limb[:,index]
        Is = stellarint(Rs,U,1000,"Nonlinear")

        print('Total stellar flux = %s' %(Is))

        if Law == "Quadratic" :

            #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
            I2 = lambda q,U : 1 - U[0]*(1 - q) - U[1]*(1 - q)**2

        if Law == "Nonlinear" :

            I2 = lambda q,U : 1 - U[0]*(1 - np.sqrt(q)) - U[1]*(1 - q) - U[2]*(1 - q**(3/2.)) - U[3]*(1 - q**2)

        gre, = np.where(sep < 1 + r)

        for i in gre :

            I_ground = np.zeros((reso_r,reso_theta))
            I_ground_plan = np.zeros((reso_occ,reso_theta))

            u_xy = - np.cos(inc_0)*np.cos(ksi[i])
            v_xy = np.sin(ksi[i])

            sep_xy = a*np.sqrt(u_xy**2 + v_xy**2)

            if sep_xy < 1 - r :

                for r_range in range(reso_r) :

                    for theta_range in range(reso_theta) :

                        y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                        x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                        pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                        #print(sep_xy,r,x,y)

                        rho = np.sqrt((sep_xy + x)**2 + y**2)
                        I_ground[r_range,theta_range] = I2(np.cos(rho),U)*pond
                        #print(I_ground[j,:])

                for r_range in range(1,reso_occ) :

                    for theta_range in range(reso_theta) :

                        y = (r_range*r/(Rs*float(reso_occ)))*np.sin(theta_range*theta_step)
                        x = (r_range*r/(Rs*float(reso_occ)))*np.cos(theta_range*theta_step)
                        pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                        #print(sep_xy,r,x,y)

                        rho = np.sqrt((sep_xy + x)**2 + y**2)
                        I_ground_plan[r_range,theta_range] = I2(np.cos(rho),U)*pond

                print("first")

            if sep_xy >= 1 - r and sep_xy <= 1 + r :

                for r_range in range(reso_r) :

                    for theta_range in range(reso_theta) :

                        y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                        x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                        pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                        rho = np.sqrt((sep_xy + x)**2 + y**2)

                        #print(r_range,theta_range,rho)

                        if rho <= 1 :

                            #print(ok)
                            I_ground[r_range,theta_range] = I2(np.cos(rho),U)*pond
                            #print(I_ground[j,:])

                for r_range in range(1,reso_occ) :

                    for theta_range in range(reso_theta) :

                        y = (r_range*r/(float(reso_occ)))*np.sin(theta_range*theta_step)
                        x = (r_range*r/(float(reso_occ)))*np.cos(theta_range*theta_step)
                        pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                        rho = np.sqrt((sep_xy + x)**2 + y**2)

                        #print(r_range,theta_range,rho)

                        if rho <= 1 :

                            I_ground_plan[r_range,theta_range] = I2(np.cos(rho),U)*pond

                print('second %s'%(sep_xy),r)

            Itot_result = Itot*I_ground
            #plt.imshow(I_ground, aspect='auto')
            #plt.colorbar()
            #plt.show()
            #plt.imshow(I_ground_plan, aspect='auto')
            #plt.colorbar()
            #plt.show()
            I_time[i_bande,i] = np.nansum(Itot_result)
            I_ground_time[i_bande,i] = np.nansum(I_ground)
            I_ground_plan_time[i_bande,i] = np.nansum(I_ground_plan)

            I_star_time[i_bande,i] = 1 - I_ground_time[i_bande,i]/Is - I_ground_plan_time[i_bande,i]/Is + I_time[i_bande,i]/Is
            print(I_star_time[i_bande,i])

        les, = np.where(sep > 1 + r)

        for i in les :

            I_star_time[i_bande,i] = 1

        #plt.imshow(I_ground,vmin=0.52,vmax=0.57)
        #plt.colorbar()
        #plt.show()

    return I_star_time


########################################################################################################################


def lightcurve_maker_opt(Rs,Rp,h,Ms,cDcorr,cPcorr,inc_0,U_limb,Itot_plan,r_step,theta_step,reso_occ,t,i_sample,bande_sample,bande_limb,Law) :

    reso_bande,reso_r,reso_theta = np.shape(Itot_plan)
    Itot = Itot_plan[i_sample,:,:]
    dim_coeff,dim_ban = np.shape(U_limb)

    I_time = np.ones((i_sample.size,t.size))
    I_ground_time = np.zeros((i_sample.size,t.size))
    I_ground_plan_time = np.zeros((i_sample.size,t.size))
    I_star_time = np.ones((i_sample.size,t.size))

    a = ((((cPcorr*24*3600)**2)*G*(Ms/(4*np.pi**2)))**(1/3.))/(Rs)
    r = (Rp + h)/Rs

    ksi = 2*np.pi*(t - cDcorr)/cPcorr
    u = - np.cos(inc_0)*np.cos(ksi)
    v = np.sin(ksi)
    sep = a*np.sqrt(u**2 + v**2)

    U = np.zeros((dim_coeff,i_sample.size))

    for i_bande in range(i_sample.size) :

        wave = 1./(bande_sample[i_sample[i_bande]]*100)*10**6
        wh, = np.where(bande_limb >= wave)

        if wh.size != 0 :

            index = wh[0] - 1

        else :

            index = bande_limb.size - 2

        U[:,i_bande] = U_limb[:,index]

    Is = stellarint_matrix(Rs,U,1000,"Nonlinear")

    print('Total stellar flux = %s' %(Is))

    if Law == "Quadratic" :

        #I2 = lambda q,U : 1 - (U[0] + 2*U[1])*(1 - np.sqrt(1 - q**2)) + U[1]*q**2
        I2 = lambda q,U : 1 - U[0,:]*(1 - q) - U[1,:]*(1 - q)**2

    if Law == "Nonlinear" :

        I2 = lambda q,U : 1 - U[0,:]*(1 - np.sqrt(q)) - U[1,:]*(1 - q) - U[2,:]*(1 - q**(3/2.)) - U[3,:]*(1 - q**2)

    gre, = np.where(sep < 1 + r)

    for i in gre :

        I_ground = np.zeros((i_sample.size,reso_r,reso_theta))
        I_ground_plan = np.zeros((i_sample.size,reso_occ,reso_theta))

        u_xy = - np.cos(inc_0)*np.cos(ksi[i])
        v_xy = np.sin(ksi[i])

        sep_xy = a*np.sqrt(u_xy**2 + v_xy**2)

        if sep_xy < 1 - r :

            for r_range in range(reso_r) :

                for theta_range in range(reso_theta) :

                    y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                    x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                    pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                    #print(sep_xy,r,x,y)

                    rho = np.sqrt((sep_xy + x)**2 + y**2)
                    I_ground[:,r_range,theta_range] = I2(np.cos(rho),U)*pond
                    #print(I_ground[j,:])

            for r_range in range(1,reso_occ) :

                for theta_range in range(reso_theta) :

                    y = (r_range*r/(Rs*float(reso_occ)))*np.sin(theta_range*theta_step)
                    x = (r_range*r/(Rs*float(reso_occ)))*np.cos(theta_range*theta_step)
                    pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                    #print(sep_xy,r,x,y)

                    rho = np.sqrt((sep_xy + x)**2 + y**2)
                    I_ground_plan[:,r_range,theta_range] = I2(np.cos(rho),U)*pond

            print("first")

        if sep_xy >= 1 - r and sep_xy <= 1 + r :

            for r_range in range(reso_r) :

                for theta_range in range(reso_theta) :

                    y = (Rp/Rs + r_range*r_step/Rs)*np.sin(theta_range*theta_step)
                    x = (Rp/Rs + r_range*r_step/Rs)*np.cos(theta_range*theta_step)
                    pond = ((Rp + (r_range + 1)*r_step)**2 - (Rp + r_range*r_step)**2)*np.pi/(float(reso_theta))

                    rho = np.sqrt((sep_xy + x)**2 + y**2)

                    #print(r_range,theta_range,rho)

                    if rho <= 1 :

                        #print(ok)
                        I_ground[:,r_range,theta_range] = I2(np.cos(rho),U)*pond
                        #print(I_ground[j,:])

            for r_range in range(1,reso_occ) :

                for theta_range in range(reso_theta) :

                    y = (r_range*r/(float(reso_occ)))*np.sin(theta_range*theta_step)
                    x = (r_range*r/(float(reso_occ)))*np.cos(theta_range*theta_step)
                    pond = (((r_range + 1)*r*Rs/(float(reso_occ)))**2 - (r_range*r*Rs/(float(reso_occ)))**2)*np.pi/(float(reso_theta))

                    rho = np.sqrt((sep_xy + x)**2 + y**2)

                    #print(r_range,theta_range,rho)

                    if rho <= 1 :

                        I_ground_plan[:,r_range,theta_range] = I2(np.cos(rho),U)*pond

            print('second %s'%(sep_xy),r)

        Itot_result = Itot*I_ground

        for i_bande in range(i_sample.size) :

            I_time[i_bande,i] = np.nansum(Itot_result[i_bande,:,:])
            I_ground_time[i_bande,i] = np.nansum(I_ground[i_bande,:,:])
            I_ground_plan_time[i_bande,i] = np.nansum(I_ground_plan[i_bande,:,:])

            I_star_time[i_bande,i] = 1 - I_ground_time[i_bande,i]/Is[i_bande] - I_ground_plan_time[i_bande,i]/Is[i_bande]\
                        + I_time[i_bande,i]/Is[i_bande]

    les, = np.where(sep > 1 + r)

    for i in les :

        I_star_time[:,i] = 1

    return I_star_time


########################################################################################################################
########################################################################################################################

"""
    TRANS2FERT1D_ALL

    Cette fonction exploite l'ensemble des outils developpes precedemment afin de produire une carte de transmittance
    dans une maille cylindrique. Si la maille n'est necessaire en soit, une seule longitude est utile dans la production
    des transits ou dans les estimations du rayon effectif (au premier ordre). Cette routine peut effectuer une
    interpolation sur les donnees, toutefois le temps de calcul est tres nettement augmente (plusieurs dizaines d'heure)
    L'utilisation des donnees brutes peut etre traite en utilisant ou non les tables dx_grid et order_grid pre-etablie.
    En fonction de la resolution initiale adoptee pour les donnees GCM, les tables dx et order permettent un petit gain
    de temps (pour les resolutions elevees).

    La production de la colonne peut etre effectuee en amont eventuellement.

    Cette fonction retourne la grille de transmittance dans une maille cylindrique I[bande,r,theta].

"""

########################################################################################################################
########################################################################################################################


def trans2fert1D_all (k_corr_data_grid,k_cont,Q_cloud,Rp,h,g0,r_step,theta_step,\
                  x_step,gauss,gauss_val,dim_bande,data,P_col,T_col,gen_col,Q_col,compo_col,ind_active,dx_grid,order_grid,pdx_grid,\
                  P_sample,T_sample,Q_sample,bande_sample,name_file,n_species,c_species,lat,long,single,\
                  bande_cloud,r_eff,r_cloud,rho_p,t,phi_rot,domain,ratio,lim_alt,rupt_alt,directory,z_grid,type,\
                  Tracer=False,Continuum=True,Isolated=False,Scattering=True,Clouds=False,Kcorr=True,Rupt=False,\
                  Middle=False,Integral=False,Module=False,Optimal=False,D3Maille=False) :

    r_size,theta_size,x_size = np.shape(dx_grid)
    number_size,z_size,lat_size,long_size = np.shape(data)

    Composition = True

    if Kcorr == True :

        k_rmd = np.zeros((T_col.size,dim_bande,gauss.size))

    else :

        k_rmd = np.zeros((T_col.size,dim_bande))

    Itot = np.ones((dim_bande,r_size,theta_size))

    bar = ProgressBar(theta_size,'Radiative transfert progression')

    for i in range(theta_size) :

        theta_line = i

        if Rupt == True :

            dep = int(rupt_alt/r_step)

            Itot[:, 0:dep, theta_line] = np.zeros((dim_bande,dep))

        else :

            dep = 0

        for j in range(dep,r_size) :

            if Middle == False :

                r = Rp + j*r_step

            else :

                r = Rp + (j+0.5)*r_step
            r_line = j

            dx = dx_grid[r_line,theta_line,:]
            order = order_grid[:,r_line,theta_line,:]
            pdx = pdx_grid[r_line,theta_line,:]

            if r < Rp + lim_alt :

                if j == 0 and i == 0 :

                    P_rmd,T_rmd,Q_rmd,k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd = \
                    convertator1D (P_col,T_col,gen_col,c_species,Q_col,compo_col,ind_active,\
                        k_corr_data_grid,k_cont,Q_cloud,P_sample,\
                        T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name_file,t,phi_rot,\
                        n_species,domain,ratio,directory,Tracer,Composition,Continuum,Scattering,Clouds,Kcorr,Optimal)

                zone, = np.where(dx >= 0)
                cut, = np.where(order[0,zone] < z_size)
                dx_ref = dx[zone[cut]]
                #data_ref = data[:,order[0,zone[cut]],order[1,zone[cut]],order[2,zone[cut]]]
                data_ref = data[:,order[0,zone[cut]],lat,long]
                P_ref, T_ref = data_ref[0], data_ref[1]

                if Tracer == True :

                    Q_ref = data_ref[2]

                    k_inter,k_cont_inter,k_sca_inter,k_cloud_inter = \
                    k_correlated_interp_remind_M(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,\
                        P_ref.size,P_rmd,P_ref,T_rmd,T_ref,Q_rmd,Q_ref,Continuum,Scattering,Clouds,Kcorr)

                else :

                    k_inter,k_cont_inter,k_sca_inter,k_cloud_inter = \
                    k_correlated_interp_remind(k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,\
                        P_ref.size,P_rmd,P_ref,T_rmd,T_ref,Continuum,Scattering,Clouds,Kcorr)

                if Module == True :
                    z_ref = z_grid[r_line,theta_line,order[3,zone[cut]]]
                    P_ref = module_density(P_ref,T_ref,z_ref,Rp,g0,data_ref[number_size-1],r_step,type,Middle)
                Cn_mol_ref = P_ref/(R_gp*T_ref)*N_A

                I_out = radiative_transfert_remind3D(dx_ref,pdx[zone[cut]],Cn_mol_ref,k_inter,k_cont_inter,k_sca_inter,\
                        k_cloud_inter,gauss_val,single,Continuum,Isolated,Scattering,Clouds,Kcorr,Integral)

                Itot[:, r_line, theta_line] = I_out[:]

        bar.animate(i)

    return Itot


########################################################################################################################
########################################################################################################################


def generator_1D_isoc(P,T,M,compo,dim_bande,Rp,h,r_step,g0,cross,ind_active, Discret=True,Integral=False,Gravity=False) :

    bar = ProgressBar(P.size-1,'Computation of the optical path')

    tau = np.zeros((dim_bande,P.size-2))
    tau3 = np.zeros((dim_bande,P.size-2))

    for i_r in range(1,P.size-1) :

        r = (i_r - 0.5)*r_step
        L = np.sqrt((Rp+h)**2 - (Rp+r)**2)
        x_pre = 0
        dist = 0

        for i_z in range(i_r,P.size-1) :

            if i_z != P.size-1 :
                z = (i_z)*r_step
                x = np.sqrt((Rp+z)**2 - (Rp+r)**2) - x_pre
                x_pre = np.sqrt((Rp+z)**2 - (Rp+r)**2)
                dist += x

                if i_z != i_r :
                    z_min = (i_z-1)*r_step
                    z_max = i_z*r_step
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                            P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*(0.5*r_step)/(1+(-0.5*r_step)/(Rp+(i_z-0.5)*r_step)))
                            g_min = g0*(1./(1.+(i_z-1)*r_step/Rp))**2
                        else :
                            P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(0.5*r_step))
                else :
                    z_min = r
                    z_max = i_z*r_step
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                            P_min = P[i_z]
                            g_min = g
                        else :
                            P_min = P[i_z]
            else :
                x = L - x_pre
                dist += x

                if i_z != i_r :
                    z_min = (i_z-1)*r_step
                    z_max = h
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                            P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*(0.5*r_step)/(1+(-0.5*r_step)/(Rp+(i_z-0.5)*r_step)))
                            g_min = g0*(1./(1.+(i_z-1)*r_step/Rp))**2
                        else :
                            P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(0.5*r_step))
                else :
                    z_min = r
                    z_max = h
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                            P_min = P[i_z]
                            g_min = g
                        else :
                            P_min = P[i_z]

            Cn = P[i_z]/(R_gp*T[i_z])*N_A

            if Discret == True :
                for i_n in range(ind_active.size) :
                    tau[:,i_r-1] += 2*x*cross[i_n]*Cn*compo[ind_active[i_n],i_z]

            if Integral == True :
                if Gravity == False :
                    INT = integrate.quad(lambda z:P_min/(R_gp*T[i_z])*np.exp(-M[i_z]*g_min/(R_gp*T[i_z])*z*(1./(1.+z/(Rp+z_min))))*(Rp+z+z_min)/(np.sqrt((Rp+z+z_min)**2-(Rp+r)**2)),0,z_max-z_min)
                else :
                    INT = integrate.quad(lambda z:P_min/(R_gp*T[i_z])*np.exp(-M[i_z]*g0/(R_gp*T[i_z])*z)*(Rp+z+z_min)/(np.sqrt((Rp+z+z_min)**2-(Rp+r)**2)),0,z_max-z_min)

                for i_n in range(ind_active.size) :
                    tau3[:,i_r-1] += 2*cross[i_n]*INT[0]*N_A*compo[ind_active[i_n],i_z]

        bar.animate(i_r)

    I_dis = np.exp(-tau)
    I_int = np.exp(-tau3)

    return I_dis, I_int

########################################################################################################################
########################################################################################################################


def generator_1D_isoc_pressure(P,T,M,compo,dim_bande,Rp,h,alt_array,g0,cross,ind_active, Discret=True,Integral=False,Gravity=False) :

    bar = ProgressBar(P.size-1,'Computation of the optical path')

    tau = np.zeros((dim_bande,P.size-2))
    tau3 = np.zeros((dim_bande,P.size-2))
    print alt_array.size, P.size

    for i_r in range(1,P.size-2) :

        r = (alt_array[i_r]+alt_array[i_r-1])/2.
        L = np.sqrt((Rp+h)**2 - (Rp+r)**2)
        x_pre = 0
        dist = 0

        for i_z in range(i_r,P.size-2) :

            if i_z != P.size-1 :
                z = alt_array[i_z]
                z_mid = (alt_array[i_z]+alt_array[i_z-1])/2.
                x_mid = (alt_array[i_z]-alt_array[i_z-1])/2.
                x = np.sqrt((Rp+z)**2 - (Rp+r)**2) - x_pre
                x_pre = np.sqrt((Rp+z)**2 - (Rp+r)**2)
                dist += x

                if i_z != i_r :
                    z_min = alt_array[i_z-1]
                    z_max = alt_array[i_z]
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+z_mid/Rp))**2
                            P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*x_mid/(1+(-x_mid)/(Rp+z_mid)))
                            g_min = g0*(1./(1.+z_min/Rp))**2
                        else :
                            P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(x_mid))
                else :
                    z_min = r
                    z_max = alt_array[i_z]
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+z_mid/Rp))**2
                            P_min = P[i_z]
                            g_min = g
                        else :
                            P_min = P[i_z]
            else :
                x = L - x_pre
                dist += x
                z_mid = (h+alt_array[i_z-1])/2.
                x_mid = (h-alt_array[i_z-1])/2.

                if i_z != i_r :
                    z_min = alt_array[i_z-1]
                    z_max = h
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+z_mid/Rp))**2
                            P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*x_mid/(1+(-x_mid)/(Rp+z_mid)))
                            g_min = g0*(1./(1.+z_min/Rp))**2
                        else :
                            P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(x_mid))
                else :
                    z_min = r
                    z_max = h
                    z_mid = r
                    if Integral == True :
                        if Gravity == False :
                            g = g0*(1./(1.+z_mid/Rp))**2
                            P_min = P[i_z]
                            g_min = g
                        else :
                            P_min = P[i_z]

            Cn = P[i_z]/(R_gp*T[i_z])*N_A

            if Discret == True :
                for i_n in range(ind_active.size) :
                    tau[:,i_r-1] += 2*x*cross[i_n]*Cn*compo[ind_active[i_n],i_z]

            if Integral == True :
                if Gravity == False :
                    INT = integrate.quad(lambda z:P_min/(R_gp*T[i_z])*np.exp(-M[i_z]*g_min/(R_gp*T[i_z])*z*(1./(1.+z/(Rp+z_min))))*(Rp+z+z_min)/(np.sqrt((Rp+z+z_min)**2-(Rp+r)**2)),0,z_max-z_min)
                else :
                    INT = integrate.quad(lambda z:P_min/(R_gp*T[i_z])*np.exp(-M[i_z]*g0/(R_gp*T[i_z])*z)*(Rp+z+z_min)/(np.sqrt((Rp+z+z_min)**2-(Rp+r)**2)),0,z_max-z_min)

                for i_n in range(ind_active.size) :
                    tau3[:,i_r-1] += 2*cross[i_n]*INT[0]*N_A*compo[ind_active[i_n],i_z]

        bar.animate(i_r)

    I_dis = np.exp(-tau)
    I_int = np.exp(-tau3)

    return I_dis, I_int


########################################################################################################################
########################################################################################################################


def generator_1D(data,n_species,m_species,c_species,dim_bande,Rp,h,r_step,g0,cross,ind_active,k_cont,\
                 Q_cloud,P_sample,T_sample,Q_sample,bande_sample,bande_cloud,r_eff,r_cloud,rho_p,name_file,t,phi_rot,\
                 domain,ratio,directory,Tracer=False,Continuum=False,Scattering=False,Clouds=False,Kcorr=False,\
                 Optimal=False,Discret=True,Integral=False,Gravity=False,Molecular=False,Pressure=True,Script=True,Middle=True) :

    P_col,T_col = data[0,:],data[1,:]
    if Tracer == True :
        Q_col = data[2,:]
    else :
        Q_col = np.array([])
    if Clouds == True :
        gen_col = data[2+m_species.size:2+m_species.size+c_species.size,:]
    else :
        gen_col = np.array([])
    compo = data[2+m_species.size+c_species.size:2+m_species.size+c_species.size+n_species.size+1,:]
    M = data[2+m_species.size+c_species.size+n_species.size,:]

    P,T,Q,cross_mol,cross_cont,cross_sca,cross_cloud = \
    convertator1D (P_col,T_col,gen_col,c_species,Q_col,compo,ind_active,\
                cross,k_cont,Q_cloud,P_sample,\
                T_sample,Q_sample,bande_sample,bande_cloud,r_step,r_eff,r_cloud,rho_p,name_file,t,phi_rot,\
                n_species,domain,ratio,directory,Tracer,Continuum,Scattering,Clouds,Kcorr,Optimal,Script)

    if Script == True :
        bar = ProgressBar(P.size-1,'Computation of the optical path')

    if Middle == True :
        tau = np.zeros((dim_bande,P.size-2))
        tau3 = np.zeros((dim_bande,P.size-2))
    else :
        tau = np.zeros((dim_bande,P.size))
        tau3 = np.zeros((dim_bande,P.size))

    for i_r in range(1,P.size-1) :

        if Middle == True :
            if Pressure == False :
                r = (i_r - 0.5)*r_step
            else :
                alt_array = r_step
                r = (alt_array[i_r]+alt_array[i_r-1])/2.
        else :
            if Pressure == False :
                r = (i_r - 1)*r_step
            else :
                alt_array = r_step
                r = alt_array[i_r-1]

        L = np.sqrt((Rp+h)**2 - (Rp+r)**2)
        x_pre = 0
        dist = 0

        for i_z in range(i_r,P.size-1) :

            if Pressure == False :

                if i_z != P.size-1 :
                    z = (i_z)*r_step
                    x = np.sqrt((Rp+z)**2 - (Rp+r)**2) - x_pre
                    x_pre = np.sqrt((Rp+z)**2 - (Rp+r)**2)
                    dist += x

                    if i_z != i_r :
                        z_min = (i_z-1)*r_step
                        z_max = i_z*r_step
                        if Integral == True :
                            if Gravity == False :
                                g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                                P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*(0.5*r_step)/(1+(-0.5*r_step)/(Rp+(i_z-0.5)*r_step)))
                                g_min = g0*(1./(1.+(i_z-1)*r_step/Rp))**2
                            else :
                                P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(0.5*r_step))
                    else :
                        z_min = r
                        z_max = i_z*r_step
                        if Integral == True :
                            if Gravity == False :
                                g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                                P_min = P[i_z]
                                g_min = g
                            else :
                                P_min = P[i_z]
                else :
                    x = L - x_pre
                    dist += x

                    if i_z != i_r :
                        z_min = (i_z-1)*r_step
                        z_max = h
                        if Integral == True :
                            if Gravity == False :
                                g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                                P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*(0.5*r_step)/(1+(-0.5*r_step)/(Rp+(i_z-0.5)*r_step)))
                                g_min = g0*(1./(1.+(i_z-1)*r_step/Rp))**2
                            else :
                                P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(0.5*r_step))
                    else :
                        z_min = r
                        z_max = h
                        if Integral == True :
                            if Gravity == False :
                                g = g0*(1./(1.+(i_z-0.5)*r_step/Rp))**2
                                P_min = P[i_z]
                                g_min = g
                            else :
                                P_min = P[i_z]

            else :

                if Middle == True :

                    if i_z != P.size-1 :
                        z = alt_array[i_z]
                        z_mid = (alt_array[i_z]+alt_array[i_z-1])/2.
                        x_mid = (alt_array[i_z]-alt_array[i_z-1])/2.
                        x = np.sqrt((Rp+z)**2 - (Rp+r)**2) - x_pre
                        x_pre = np.sqrt((Rp+z)**2 - (Rp+r)**2)
                        dist += x

                        if i_z != i_r :
                            z_min = alt_array[i_z-1]
                            z_max = alt_array[i_z]
                            if Integral == True :
                                if Gravity == False :
                                    g = g0*(1./(1.+z_mid/Rp))**2
                                    P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*x_mid/(1+(-x_mid)/(Rp+z_mid)))
                                    g_min = g0*(1./(1.+z_min/Rp))**2
                                else :
                                    P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(x_mid))
                        else :
                            z_min = r
                            z_max = alt_array[i_z]
                            if Integral == True :
                                if Gravity == False :
                                    g = g0*(1./(1.+z_mid/Rp))**2
                                    P_min = P[i_z]
                                    g_min = g
                                else :
                                    P_min = P[i_z]
                    else :
                        x = L - x_pre
                        dist += x
                        z_mid = (h+alt_array[i_z-1])/2.
                        x_mid = (h-alt_array[i_z-1])/2.

                        if i_z != i_r :
                            z_min = alt_array[i_z-1]
                            z_max = h
                            if Integral == True :
                                if Gravity == False :
                                    g = g0*(1./(1.+z_mid/Rp))**2
                                    P_min = P[i_z]*np.exp(M[i_z]*g/(R_gp*T[i_z])*x_mid/(1+(-x_mid)/(Rp+z_mid)))
                                    g_min = g0*(1./(1.+z_min/Rp))**2
                                else :
                                    P_min = P[i_z]*np.exp(M[i_z]*g0/(R_gp*T[i_z])*(x_mid))
                        else :
                            z_min = r
                            z_max = h
                            z_mid = r
                            if Integral == True :
                                if Gravity == False :
                                    g = g0*(1./(1.+z_mid/Rp))**2
                                    P_min = P[i_z]
                                    g_min = g
                                else :
                                    P_min = P[i_z]

                else :

                    if i_z != P.size-1 :
                        z = alt_array[i_z-1]
                        x = np.sqrt((Rp+z)**2 - (Rp+r)**2) - x_pre
                        x_pre = np.sqrt((Rp+z)**2 - (Rp+r)**2)
                        dist += x

                        z_min = alt_array[i_z-1]
                        z_max = alt_array[i_z]
                        if Integral == True :
                            if Gravity == False :
                                P_min = P[i_z-1]
                                g_min = g0*(1./(1.+z_min/Rp))**2
                            else :
                                P_min = P[i_z-1]

            Cn = P[i_z]/(R_gp*T[i_z])*N_A

            if Discret == True :
                if Molecular == True :
                    tau[:,i_r-1] += 2*x*cross_mol[i_z]*Cn
                if Continuum == True :
                    tau[:,i_r-1] += 2*x*cross_cont[i_z]
                if Scattering == True :
                    tau[:,i_r-1] += 2*x*cross_sca[i_z]
                if Clouds == True :
                    for i_c in range(c_species.size) :
                        tau[:,i_r-1] += 2*x*cross_cloud[i_c,i_z]

            if Integral == True :
                if Gravity == False :
                    INT = integrate.quad(lambda z:P_min/(R_gp*T[i_z])*np.exp(-M[i_z]*g_min/(R_gp*T[i_z])*z*(1./(1.+z/(Rp+z_min))))*(Rp+z+z_min)/(np.sqrt((Rp+z+z_min)**2-(Rp+r)**2)),0,z_max-z_min)
                else :
                    INT = integrate.quad(lambda z:P_min/(R_gp*T[i_z])*np.exp(-M[i_z]*g0/(R_gp*T[i_z])*z)*(Rp+z+z_min)/(np.sqrt((Rp+z+z_min)**2-(Rp+r)**2)),0,z_max-z_min)
                if Molecular == True :
                    tau3[:,i_r-1] += 2*cross_mol[i_z]*INT[0]*N_A
                if Continuum == True :
                    tau3[:,i_r-1] += 2*x*cross_cont[i_z]
                if Scattering == True :
                    tau3[:,i_r-1] += 2*x*cross_sca[i_z]
                if Clouds == True :
                    for i_c in range(c_species.size) :
                        tau3[:,i_r-1] += 2*x*cross_cloud[i_c,i_z]

        if Script == True :
            bar.animate(i_r)

    I_dis = np.exp(-tau)
    I_int = np.exp(-tau3)

    return I_dis, I_int


########################################################################################################################
########################################################################################################################


def spectral_distribution(nest_out_path,output_file,dimension,live_number,Mp,Rs,P_surf,P_h,n_layers,n_species,number,\
            m_species,c_species,cross,ind_active,k_cont,Q_cloud,P_sample,T_sample,\
            Q_sample,bande_sample,bande_cloud,r_eff,r_cloud,rho_p,name_file,domain,path,x_species_inactive,\
            Tracer,Composition,Continuum,Scattering,Clouds,Kcorr,Optimal,Discret,Integral,Gravity,Molecular) :

    dim_bande = bande_sample.size
    nest_out_summary = '%s1-stats.dat'%(nest_out_path)
    data = open(nest_out_summary,'r')
    summ = data.readlines()
    x_ref = np.zeros(ind_active.size)
    for i_n in range(ind_active.size) :
        x_ref[i_n] = np.float(summ[13+dimension+3+i_n][7:32])

    wh_N2, = np.where(n_species == 'N2')
    if wh_N2.size != 0 :
        x_ref_init = x_ref
        x_ref = np.zeros(n_species.size-2)
        dec = 0
        for i_n in range(n_species.size-2) :
            if i_n == wh_N2[0]-2 :
                x_ref[i_n] = np.log10(x_species_inactive[0])
                dec = 1
            else :
                x_ref[i_n] = x_ref_init[i_n-dec]
    T_iso_ref = np.float(summ[13+dimension+i_n+3][7:32])
    Rp_ref = np.float(summ[13+dimension+4+i_n][7:32])*R_J

    sys.path.append('/Users/caldas/Desktop/Pytmosph3R/Tools/')
    from Tool_pressure_generator import pressure_scaling_taurex

    x_ratio_species_active = np.array([10**x_ref])
    M_species, M, x_ratio_species = ratio(n_species,x_ratio_species_active,IsoComp=True)
    data,alt_array,alt_mid,H_array,g_array,h,delta_z,r_step,x_step,theta_number,reso_alt,z_lim,z_reso,z_array = \
    pressure_scaling_taurex(T_iso_ref,Rp_ref,M,Mp,P_surf,P_h,n_layers,n_species,number,x_ratio_species,False,False)

    r_step = alt_array
    g0 = g_array[0]

    I_dis, I_int = generator_1D(data,n_species,m_species,c_species,dim_bande,Rp_ref,h,r_step,g0,cross,ind_active,k_cont,\
            Q_cloud,P_sample,T_sample,Q_sample,bande_sample,bande_cloud,r_eff,r_cloud,rho_p,name_file,0,0.00,\
            domain,ratio_HeH2,path,Tracer,Composition,Continuum,Scattering,Clouds,Kcorr,\
            Optimal,Discret,Integral,Gravity,Molecular,True,False)

    Alt = np.zeros(n_layers+2)
    I = np.zeros((dim_bande,n_layers+2))
    Alt[0] = 0
    Alt[n_layers+1] = alt_array[n_layers-1]
    I[:,n_layers+1] = I_dis[:,n_layers-1]
    Alt[1:n_layers+1] = alt_mid
    I[:,1:n_layers+1] += I_dis
    flux_ref = np.zeros(dim_bande)
    for i_b in range(dim_bande) :
        x = Alt
        alpha = 2*integrate.trapz((Rp_ref+x)*(1-I[i_b]),x)
        flux_ref[i_b] = (Rp_ref**2+alpha)/Rs**2

    phys = np.zeros((2,dimension),dtype=np.float64)

    for i_dim in range(dimension) :
        phys[0,i_dim] = np.float(summ[13+i_dim][7:32]) - np.float(summ[13+i_dim][36:60])
        phys[1,i_dim] = np.float(summ[13+i_dim][7:32]) + np.float(summ[13+i_dim][36:60])

    nest_out_file = '%s1-phys_live.points'%(nest_out_path)
    data = open(nest_out_file,'r')
    points = data.readlines()

    phys_nest = np.zeros((live_number,dimension+1))

    for i_num in range(live_number) :
        no = 0
        for i_dim in range(dimension) :
            i_deb = np.int(3+i_dim*28)
            i_fin = np.int(3+(i_dim+1)*28)
            if np.float(points[i_num][i_deb:i_fin]) <= phys[0,i_dim] or np.float(points[i_num][i_deb:i_fin]) >= phys[1,i_dim] :
                no = 1
        if no == 0 :
            for i_dim in range(dimension) :
                i_deb = np.int(3+i_dim*28)
                i_fin = np.int(3+(i_dim+1)*28)
                phys_nest[i_num,i_dim] = np.float(points[i_num][i_deb:i_fin])

    wh_nz, = np.where(phys_nest[:,0] != 0)
    phys_nest = phys_nest[wh_nz,:]

    live_point = wh_nz.size
    phys_spectrum = np.zeros((live_point,dim_bande))

    bar = ProgressBar(live_point,'Build of the spectral distribution : ')

    for i_p in range(live_point) :
        x_ratio_species_active = 10**phys_nest[i_p,:ind_active.size]
        if wh_N2.size != 0 :
            x_ref_init = x_ratio_species_active
            x_ratio_species_active = np.zeros(n_species.size-2)
            dec = 0
            for i_n in range(n_species.size-2) :
                if i_n == wh_N2[0]-2 :
                    x_ratio_species_active[i_n] = x_species_inactive[0]
                    dec = 1
                else :
                    x_ratio_species_active[i_n] = x_ref_init[i_n-dec]
        M_species, M, x_ratio_species = ratio(n_species,x_ratio_species_active,IsoComp=True)
        M = 0.00281
        T_iso = phys_nest[i_p,ind_active.size]
        Rp = phys_nest[i_p,ind_active.size+1]*R_J

        data,alt_array,alt_mid,H_array,g_array,h,delta_z,r_step,x_step,theta_number,reso_alt,z_lim,z_reso,z_array = \
        pressure_scaling_taurex(T_iso,Rp,M,Mp,P_surf,P_h,n_layers,n_species,number,x_ratio_species,False,False)

        r_step = alt_array
        g0 = g_array[0]

        I_dis, I_int = generator_1D(data,n_species,m_species,c_species,dim_bande,Rp,h,r_step,g0,cross,ind_active,k_cont,\
                Q_cloud,P_sample,T_sample,Q_sample,bande_sample,bande_cloud,r_eff,r_cloud,rho_p,name_file,0,0.00,\
                domain,ratio_HeH2,path,Tracer,Composition,Continuum,Scattering,Clouds,Kcorr,\
                Optimal,Discret,Integral,Gravity,Molecular,True,False)

        Alt = np.zeros(n_layers+2)
        I = np.zeros((dim_bande,n_layers+2))
        Alt[0] = 0
        Alt[n_layers+1] = alt_array[n_layers-1]
        I[:,n_layers+1] = I_dis[:,n_layers-1]
        Alt[1:n_layers+1] = alt_mid
        I[:,1:n_layers+1] += I_dis
        flux_dis = np.zeros(dim_bande)
        for i_b in range(dim_bande) :
            x = Alt
            alpha = 2*integrate.trapz((Rp+x)*(1-I[i_b]),x)
            flux_dis[i_b] = (Rp**2+alpha)/Rs**2

        phys_spectrum[i_p,:] = flux_dis

        bar.animate(i_p+1)

    wh_ri, = np.where(bande_sample!=0)
    wl_ref = 1./bande_sample*10000.
    wl = 1./bande_sample[wh_ri]*10000.
    phys_spectrum = phys_spectrum[:,wh_ri]

    dim_bande = wl.size

    plt.figure()
    plt.semilogx()
    plt.grid(True)
    for i_p in range(live_point) :
        plt.plot(wl,phys_spectrum[i_p,:],'r-')
    plt.draw()

    distrib_spectrum = np.zeros((2,dim_bande))

    for i_bande in range(dim_bande) :
        wh_min, = np.where(phys_spectrum[:,i_bande] == np.amin(phys_spectrum[:,i_bande]))
        wh_max, = np.where(phys_spectrum[:,i_bande] == np.amax(phys_spectrum[:,i_bande]))

        distrib_spectrum[0,i_bande] = phys_spectrum[wh_min[0],i_bande]
        distrib_spectrum[1,i_bande] = phys_spectrum[wh_max[0],i_bande]

    data = pickle.load(open('%sOutput/%s/nest_out.pickle'%(path,output_file),'rb'))
    data.keys()

    wvl = data['solutions'][0]['obs_spectrum'][:,0]
    flux_obs = data['solutions'][0]['obs_spectrum'][:,1]
    wvl_fit = data['solutions'][0]['fit_spectrum_xsecres'][:,0]
    flux_fit = data['solutions'][0]['fit_spectrum_xsecres'][:,1]

    plt.figure()
    plt.semilogx()
    plt.grid(True)
    plt.errorbar(wvl,flux_obs, yerr=5.e-5,color='k')
    plt.plot(wvl_fit,flux_fit,'+-')
    plt.fill_between(wl,distrib_spectrum[1],facecolor='blue')
    plt.fill_between(wl,distrib_spectrum[0],facecolor='white')
    plt.plot(wl_ref,flux_ref,'+-y')
    plt.axis([0.3,50,np.amin(flux_fit),np.amax(flux_fit)])
    plt.ylabel('Relative flux')
    plt.xlabel('Wavelength (micron)')
    plt.draw()


########################################################################################################################


def parameters_signature(I,theta,theta_number,crit,bande_sample,Rs,Rp,r_step,n_layers,data_convert,reso_long,reso_lat,reso_alt,\
                phi_obli,phi_rot,path,name_file,stitch_file,Kcorr,Middle,D3maille=False) :

    dim_bande = bande_sample.size

    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I,bande_sample,Rs,Rp,r_step,0,\
                                                                                False,Kcorr,Middle)
    bande_max, = np.where(flux == np.amax(flux))
    bande_min, = np.where(flux == np.amin(flux))
    flux_max = flux[bande_max]
    flux_min = flux[bande_min]

    sign = np.zeros((n_layers,dim_bande))
    sign_up = np.zeros((n_layers,dim_bande))
    sign_down = np.zeros((n_layers,dim_bande))
    ind = np.zeros((n_layers,dim_bande),dtype='int')
    bar = ProgressBar(dim_bande,'Progression : ')
    for i_bande in range(dim_bande) :
        stop = 0
        init = 0
        for i_r in range(1,n_layers) :
            if stop == 0 and init == 1 and I[i_bande,n_layers-i_r,theta] <= 0.999 :
                I_s = np.ones((1,n_layers,1))
                I_s[0,:n_layers-i_r,0] = I[i_bande,:n_layers-i_r,theta]
                R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I_s,np.array([1]),Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)
                contrib = np.abs(flux - flux_ref)/flux_ref
                flux_ref = flux
                if contrib <= crit :
                    sign_up[n_layers-i_r-1,i_bande] = 0.
                    ind[n_layers-i_r-1,i_bande] = 0
                    stop = 1
                else :
                    sign_up[n_layers-i_r-1,i_bande] = np.abs(contrib)*100.
                    ind[n_layers-i_r-1,i_bande] = i_r

            if stop == 0 and init == 0 and I[i_bande,n_layers-i_r,theta] >= 0.001 :
                I_s = np.ones((1,n_layers,1))
                I_s[0,:n_layers-i_r,0] = I[i_bande,:n_layers-i_r,theta]
                R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I_s,np.array([1]),Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)
                if i_r == 1 :
                    I_s = np.zeros((1,n_layers,1))
                    I_s[0,:,0] = I[i_bande,:n_layers,theta]
                    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux_ref = atmospectre(I_s,np.array([1]),Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)
                contrib = np.abs(flux - flux_ref)/flux_ref
                flux_ref = flux
                if contrib <= crit :
                    sign_up[n_layers-i_r-1,i_bande] = 0.
                    ind[n_layers-i_r-1,i_bande] = 0
                else :
                    sign_up[n_layers-i_r-1,i_bande] = np.abs(contrib)*100.
                    ind[n_layers-i_r-1,i_bande] = i_r
                    init = 1
        stop = 0
        init = 0
        for i_r in range(1,n_layers) :
            if stop == 0 and init == 1 and I[i_bande,i_r,theta] >= 0.001:
                I_s = np.zeros((1,n_layers,1))
                I_s[0,:i_r,0] = I[i_bande,:i_r,theta]
                R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I_s,np.array([1]),Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)
                contrib = np.abs(flux - flux_ref)/flux_ref
                flux_ref = flux
                if contrib <= crit :
                    sign_down[i_r,i_bande] = 0.
                    ind[i_r,i_bande] = 0
                    stop = 1
                else :
                    sign_down[i_r,i_bande] = np.abs(contrib)*100.
                    ind[i_r,i_bande] = i_r

            if stop == 0 and init == 0 and I[i_bande,i_r,theta] <= 0.999 :
                I_s = np.zeros((1,n_layers,1))
                I_s[0,:i_r,0] = I[i_bande,:i_r,theta]
                R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(I_s,np.array([1]),Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)
                if i_r == 1 :
                    I_s = np.zeros((1,n_layers,1))
                    I_s[0,0,0] = I[i_bande,0,theta]
                    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux_ref = atmospectre(I_s,np.array([1]),Rs,Rp,r_step,0,\
                                                                                        False,Kcorr,Middle)
                contrib = np.abs(flux - flux_ref)/flux_ref
                flux_ref = flux
                if contrib <= crit :
                    sign_down[i_r,i_bande] = 0.
                    ind[i_r,i_bande] = 0
                else :
                    sign_down[i_r,i_bande] = np.abs(contrib)*100.
                    ind[i_r,i_bande] = i_r
                    init = 1

        for i_r in range(n_layers-1) :
            sign[i_r,i_bande] = np.amin([sign_up[i_r,i_bande],sign_down[i_r,i_bande]])
        bar.animate(i_bande+1)

    wh_ze, = np.where(bande_sample != 0)
    wl = np.log10(1./bande_sample[wh_ze]*10000.)
    x = np.linspace(0,100,n_layers)
    y = np.linspace(np.amin(wl),np.amax(wl),100000)
    def f(X,Y,S,wl):
        P = np.zeros((X.size,Y.size))
        for i in range(Y.size) :
            wh, = np.where(wl <= Y[i])
            P[:,i] = S[:,wh[0]]
        return P
    plt.figure(4)
    if D3maille == False :
        plt.imshow(f(x,y,sign,wl),origin='lower',aspect='auto',extent=[np.amin(wl), np.amax(wl),\
        np.log10(data_convert[0,0,0,reso_lat/2,reso_long/2]/100.),np.log10(data_convert[0,0,n_layers+1,reso_lat/2,reso_long/2]/100.)],cmap='hot')
    else :
        plt.imshow(f(x,y,sign,wl),origin='lower',aspect='auto',extent=[np.amin(wl), np.amax(wl),\
        np.log10(data_convert[0,0,0,reso_lat/2,reso_long/2]/100.),np.log10(data_convert[0,0,n_layers,reso_lat/2,reso_long/2]/100.)],cmap='hot')
    plt.ylabel('Pressure (mBar)')
    plt.xlabel('Wavelenth (power of 10, micron)')
    plt.colorbar()

    wh_max, = np.where(ind[:,bande_max[0]] != 0)
    ind_max = ind[wh_max,bande_max]
    wh_min, = np.where(ind[:,bande_min[0]] != 0)
    ind_min = ind[wh_min,bande_min]

    order_grid = np.load("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,\
                reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
    dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))

    plt.figure(5)
    plt.grid(True)
    for i_ind in ind_max :
        wh_nu, = np.where(dx_grid[i_ind+1,theta,:] > 0)
        dx = np.zeros(wh_nu.size)
        for i_nu in range(wh_nu.size) :
            dx[i_nu] = np.nansum(dx_grid[i_ind,theta,wh_nu[:i_nu+1]])
        plt.plot(dx/1000.,\
            data_convert[1,0,order_grid[0,i_ind+1,theta,wh_nu],order_grid[1,i_ind+1,theta,wh_nu],order_grid[2,i_ind+1,theta,wh_nu]],'r-',linewidth=4)

    for i_ind in ind_min :
        wh_nu, = np.where(dx_grid[i_ind+1,theta,:] > 0)
        dx = np.zeros(wh_nu.size)
        for i_nu in range(wh_nu.size) :
            dx[i_nu] = np.nansum(dx_grid[i_ind,theta,wh_nu[:i_nu+1]])
        plt.plot(dx/1000.,\
            data_convert[1,0,order_grid[0,i_ind+1,theta,wh_nu],order_grid[1,i_ind+1,theta,wh_nu],order_grid[2,i_ind+1,theta,wh_nu]],'g-',linewidth=4)
    plt.ylabel('Temperature (K)')
    plt.xlabel('Position (km)')
    plt.draw()