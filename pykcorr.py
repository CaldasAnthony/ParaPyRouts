
from pytransfert import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
number_rank = comm.size

########################################################################################################################
########################################################################################################################

"""
    PYKCORR

    Cette bibliotheque contient l'ensemble des routines permettant l'interpolation des sections efficaces ou des
    k coefficients dans les atmospheres qui nous interessent. Certaines fonctions sont dupliquees en fonction de la
    presence ou non d'un marqueur et de l'utilisation de sections efficaces plutot que de k coefficients. Ces routines
    sont executees par le script Parameters (qui peut etre precede d'un acronyme de l'exoplanete etudiee, par exemple :
    GJParameters).

    Les tableaux d'opacites ainsi generes sont directement utilises dans les modules de pytransfert et de pyremind afin
    de resoudre le transfert radiatif. Certaines fonctions sont utilisees directement dans la routine de transfert 1D.
    La cle de voute de cette bibliotheque, a savoir convertator, permet la generation de l'ensemble des opacites
    (moleculaire, continuum, diffusion Rayleigh et diffusion de Mie)

    Version : 5.6

    Recentes mise a jour :

    >> Modification totale de l'identification des couples P,T ou P,T,Q dans les simulations 3D dans les routines
    convertator
    >> Correction d'un bug de cloud_scattering qui faisait que la meme masse molaire etait adoptee pour toutes les
    couches de l'atmosphere, a present la routine tient compte de la diversite sur le poids moleculaire moyen dans les
    differentes cellules associees aux couples P,T ou P,T,Q

    Date de derniere modification : 10.10.2016

    >> Reecriture du calcul des donnees continuum, elle s'adapte desormais a une plus grande diversite de sources
    continuum (l'eau, le methane, le dioxyde de carbone ...), et utilise les fonctions deja existante pour calculer
    les aspects self et foreign

    Date de derniere modification : 29.10.2017

    >> Correction d'un bug qui faisait qu'on ne prenait pas a la fois les donnees H2-H2 et H2-He lors du calcul du
    continuum. Reecriture egalement pour permettre de travailler avec des atmopshere ne comportant pas soit l'un soit
    l'autre.
    >> Optimisation de l'ecriture et de la methode d'extraction des donnnees continuum toujours.
    >> Correction d'une partie du code d'interpolation dans les donnees continuum.
    >> Correction de la methode de calcul des opacites continuum ... oui il y avait un probleme avec le continuum de
    toute evidence ^^

    Date de derniere modification : 06.12.2017

    >> Refonte complete de la fonction k_correlated_interp. De part son ecriture, il semblerait que des indices se soient
    melanges et introduisait une erreur sur l'interpolation en temperature et en pression pour les parties en dehors de
    la grille, raison pour laquelle les spectres ne ressemblaient pas ce qu'on attendait pour les hautes tempertaures
    pour lesquelles la region d'inversion de la profondeur optique se produisait pour des pressions en dehors de la grille
    >> Correction de trois bugs majeurs dans Ssearcher qui faisait que meme apres correction de k_correlated_interp, nous
    ne produisions pas des spectres satisfaisants. Des i_Tu transformes en i_Td, des c13 appliques a la mauvaise section
    efficace ... tous ces effets etaient compenses par l'erreur dans k_correlated (sauf dans le cas des hautes temperatures)
    >> Reecriture globale des fonctions d'interpolation.

    Date de derniere modification : 05.03.2018

"""

########################################################################################################################
########################################################################################################################

def convertator_save(P_rmd,T_rmd,rmind,Q_rmd,gen_cond_rmd,composit_rmd,directory,name,reso_long,reso_lat,name_exo,t,\
                    x_step,phi_rot,phi_obli,domain,dim_bande,dim_gauss,rank,Kcorr=True,Tracer=False,Clouds=False,ByLay=False) :

    if ByLay == True :
        domain += '_%i'%(rank)
        name += '/Temp'

    if Kcorr == True :
        np.save("%s%s/P_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),P_rmd)
        np.save("%s%s/T_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),T_rmd)
        np.save("%s%s/rmind_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),rmind)
        if Tracer == True :
            np.save("%s%s/Q_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),Q_rmd)
    else :
        np.save("%s%s/P_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),P_rmd)
        np.save("%s%s/T_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),T_rmd)
        np.save("%s%s/rmind_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy" \
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),rmind)
        if Tracer ==True :
            np.save("%s%s/Q_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),Q_rmd)

    if Clouds == True :

        if Kcorr == True :
            np.save("%s%s/gen_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),gen_cond_rmd)
        else :
            np.save("%s%s/gen_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),gen_cond_rmd)

    if Kcorr == True :
        np.save("%s%s/compo_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),composit_rmd)
    else :
        np.save("%s%s/compo_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),composit_rmd)


########################################################################################################################


def convertator (P_rmd,T_rmd,gen_cond_rmd,c_species,Q_rmd,composit_rmd,ind_active,ind_cross,K,K_cont,Qext,P_sample,T_sample,\
                 Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name,t,phi_rot,phi_obli,n_species,domain,ratio,directory,name_exo,reso_long,reso_lat,\
                 rank,rank_ref,rank_max,name_source,Tracer=False,Molecular=False,Continuum=False,Clouds=False,Scattering=False,Kcorr=True,Optimal=False,ByLay=False) :

    if rank_max != comm.size :
        number_rank = rank_max
    else:
        number_rank = comm.size
    if ByLay == True :
        domain += '_%i'%(rank)
        name += '/Temp'

    zero, = np.where(P_rmd == 0.)
    
    if Molecular == True :
    
        K = np.load(K)

        if Kcorr == True :
            if Tracer == True :
                t_size,p_size,q_size,dim_bande,dim_gauss = np.shape(K)
            else :
                t_size,p_size,dim_bande,dim_gauss = np.shape(K)
        else :
            K = K[ind_cross]
            np_size,p_size,t_size,dim_bande = np.shape(K)

        if Kcorr == True :
            if Tracer == False :
                k_rmd = Ksearcher(T_rmd,P_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal)
            else :
                k_rmd = Ksearcher_M(T_rmd,P_rmd,Q_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,Optimal)

            if rank_max == comm.size :
                comm.Barrier()

            if ByLay == False :
                if rank != 0 :
                    sh_k = np.array(np.shape(k_rmd),dtype=np.int)
                    comm.Send([sh_k,MPI.INT],dest=0,tag=0)
                    comm.Send([k_rmd,MPI.DOUBLE],dest=0,tag=1)
                if rank == 0 :
                    k_rmd_tot = k_rmd
                    print "Reconstruction of molecular absorptions will begin"
                    bar = ProgressBar(number_rank,"Reconstruction of k-correlated absorptions")
                    for i_n in range(1,number_rank) :
                        sh_k = np.zeros(2,dtype=np.int)
                        comm.Recv([sh_k,MPI.INT],source=i_n,tag=0)
                        k_rmd_n = np.zeros((sh_k),dtype=np.float64)
                        comm.Recv([k_rmd_n,MPI.DOUBLE],source=i_n,tag=1)
                        k_rmd_tot = np.concatenate((k_rmd_tot,k_rmd_n))
                        bar.animate(i_n+1)

                    np.save("%s%s/k_corr_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_rmd_tot)
                    del k_rmd_tot
            else :
                np.save("%s%s/k_corr_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_rmd)

        else :
            compo_active = composit_rmd[ind_active,:]
            if Tracer == False :
                k_rmd = Ssearcher(T_rmd,P_rmd,compo_active,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal)
            else :
                k_rmd = Ssearcher(T_rmd,P_rmd,compo_active,K,P_sample,T_sample,rank,rank_ref,Kcorr,Optimal)

            if rank_max == comm.size :
                comm.Barrier()

            if ByLay == False :
                if rank != 0 :
                    sh_k = np.array(np.shape(k_rmd),dtype=np.int)
                    comm.Send([sh_k,MPI.INT],dest=0,tag=0)
                    comm.Send([k_rmd,MPI.DOUBLE],dest=0,tag=1)
                if rank == 0 :
                    k_rmd_tot = k_rmd
                    if Optimal == False :
                        print "Reconstruction of molecular absorptions will begin"
                        bar = ProgressBar(number_rank,"Reconstruction of molecular absorptions")
                    else :
                        print "Reconstruction of optimized molecular absorptions will begin"
                        bar = ProgressBar(number_rank,"Reconstruction of optimized molecular absorptions")
                    for i_n in range(1,number_rank) :
                        sh_k = np.zeros(2,dtype=np.int)
                        comm.Recv([sh_k,MPI.INT],source=i_n,tag=0)
                        k_rmd_n = np.zeros((sh_k),dtype=np.float64)
                        comm.Recv([k_rmd_n,MPI.DOUBLE],source=i_n,tag=1)
                        k_rmd_tot = np.concatenate((k_rmd_tot,k_rmd_n))
                        bar.animate(i_n+1)

                    if Optimal == False :
                        np.save("%s%s/k_cross_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_rmd_tot)
                    else :
                        np.save("%s%s/k_cross_opt_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                        %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_rmd_tot)

                    del k_rmd_tot
            else :
                if Optimal == False :
                    np.save("%s%s/k_cross_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_rmd)
                else :
                    np.save("%s%s/k_cross_opt_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_rmd)

            if rank == 0 :
                if Kcorr == True :
                    print "Ksearcher finished with success"
                else :
                    print "Ssearcher finished with success"
        del k_rmd,K

    else :

        del K

    if rank_max == comm.size :
        comm.Barrier()

    if Continuum == True :

        if Kcorr == True :
            dim_bande = bande_sample.size-1
        else :
            dim_bande = bande_sample.size
        cont_species = K_cont.species
        H2, He, Other = H2HeO(cont_species)
        amagat = 2.69578e-3*P_rmd/T_rmd
        k_cont_rmd = np.zeros((dim_bande,P_rmd.size))
        decont = 0

        if H2 == True :

            decont += 1
            K_cont_h2h2 = np.load('%s%s/k_cont_%s.npy'%(directory,name_source,K_cont.associations[0]))
            K_cont_nu_h2h2 = np.load('%s%s/k_cont_nu_%s.npy'%(directory,name_source,K_cont.associations[0]))
            T_cont_h2h2 = np.load('%s%s/T_cont_%s.npy'%(directory,name_source,K_cont.associations[0]))

            k_interp_h2h2 = k_cont_interp_h2h2_integration(K_cont_h2h2,K_cont_nu_h2h2,\
                                            T_rmd,bande_sample,T_cont_h2h2,rank,rank_ref,Kcorr)


            amagat_h2h2 = amagat*composit_rmd[0,:]

            for i_bande in range(dim_bande) :
                k_cont_rmd[i_bande,:] = amagat_h2h2**2*k_interp_h2h2[i_bande,:]

            del amagat_h2h2,k_interp_h2h2,K_cont_h2h2

        if He == True :

            decont += 1
            K_cont_h2he = np.load('%s%s/k_cont_%s.npy'%(directory,name_source,K_cont.associations[1]))
            K_cont_nu_h2he = np.load('%s%s/k_cont_nu_%s.npy'%(directory,name_source,K_cont.associations[1]))
            T_cont_h2he = np.load('%s%s/T_cont_%s.npy'%(directory,name_source,K_cont.associations[1]))


            k_interp_h2he = k_cont_interp_h2he_integration(K_cont_h2he,K_cont_nu_h2he,\
                                            T_rmd,bande_sample,T_cont_h2he,rank,rank_ref,Kcorr)

            amagat_self = amagat*composit_rmd[0,:]
            amagat_foreign = amagat*composit_rmd[1,:]

            for i_bande in range(dim_bande) :
                k_cont_rmd[i_bande,:] += amagat_foreign*amagat_self*k_interp_h2he[i_bande,:]

            del amagat_foreign,k_interp_h2he,K_cont_h2he

        if Other == False :
            del amagat
        else :
            for i_cont in range(decont,cont_species.size) :

                K_cont_spespe = np.load('%s%s/k_cont_%s.npy'%(directory,name_source,K_cont.associations[i_cont]))
                K_cont_nu_spespe = np.load('%s%s/k_cont_nu_%s.npy'%(directory,name_source,K_cont.associations[i_cont]))
                T_cont_spespe = np.load('%s%s/T_cont_%s.npy'%(directory,name_source,K_cont.associations[i_cont]))

                if cont_species[i_cont] != 'H2O' and cont_species[i_cont] != 'H2Os':
                    wh_c, = np.where(n_species == cont_species[i_cont])
                    amagat_spefor = amagat*composit_rmd[0,:]
                    amagat_speself = amagat*composit_rmd[wh_c[0],:]
                    amagat_spe = amagat_spefor*amagat_speself
                else :
                    wh_c, = np.where(n_species == 'H2O')
                    H2O = True
                    N_mol = P_rmd/(k_B*T_rmd)
                    if cont_species[i_cont] == 'H2O' :
                        amagat_spe = amagat*(1.-composit_rmd[wh_c[0],:])*composit_rmd[wh_c[0],:]*N_mol
                    if cont_species[i_cont] == 'H2Os' :
                        amagat_spe = amagat*composit_rmd[wh_c[0],:]**2*N_mol

                k_interp_spespe = k_cont_interp_spespe_integration(K_cont_spespe,K_cont_nu_spespe,\
                                T_rmd,bande_sample,T_cont_spespe,rank,rank_ref,K_cont.associations[i_cont],Kcorr,H2O)

                for i_bande in range(dim_bande) :

                    k_cont_rmd[i_bande,:] += amagat_spe*k_interp_spespe[i_bande,:]

                del amagat_spe,k_interp_spespe

        if rank_max == comm.size :
            comm.Barrier()

        if ByLay == False :

            if rank != 0 :
                sh_k = np.array(np.shape(k_cont_rmd),dtype=np.int)
                comm.Send([sh_k,MPI.INT],dest=0,tag=0)
                comm.Send([k_cont_rmd,MPI.DOUBLE],dest=0,tag=1)
            if rank == 0 :
                bar = ProgressBar(number_rank,"Reconstruction of continuum absorptions will begin")
                k_cont_rmd_tot = k_cont_rmd
                for i_n in range(1,number_rank) :
                    sh_k = np.zeros(2,dtype=np.int)
                    comm.Recv([sh_k,MPI.INT],source=i_n,tag=0)
                    k_cont_rmd_n = np.zeros((sh_k),dtype=np.float64)
                    comm.Recv([k_cont_rmd_n,MPI.DOUBLE],source=i_n,tag=1)
                    k_cont_rmd_tot = np.concatenate((k_cont_rmd_tot,k_cont_rmd_n),axis=1)
                    bar.animate(i_n+1)

                if Kcorr == True :
                    np.save("%s%s/k_cont_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),np.transpose(k_cont_rmd_tot))
                else :
                    np.save("%s%s/k_cont_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),np.transpose(k_cont_rmd_tot))

            del k_cont_rmd_tot
        else :
            if Kcorr == True :
                np.save("%s%s/k_cont_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),np.transpose(k_cont_rmd))
            else :
                np.save("%s%s/k_cont_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),np.transpose(k_cont_rmd))

        if rank == 0 :
                print "Integration of the continuum finished with success"
        del k_cont_rmd

    else :

        if rank == rank_ref :
            print "There is no continuum"

    if rank_max == comm.size :
        comm.Barrier()

    x_mol_species = composit_rmd[0:n_species.size,:]

    if Scattering == True :

        k_sca_rmd = Rayleigh_scattering(P_rmd,T_rmd,bande_sample,x_mol_species,n_species,zero,rank,rank_ref,Kcorr)

        if rank_max == comm.size :
            comm.Barrier()

        if ByLay == False :
            if rank != 0 :
                sh_k = np.array(np.shape(k_sca_rmd),dtype=np.int)
                comm.Send([sh_k,MPI.INT],dest=0,tag=0)
                comm.Send([k_sca_rmd,MPI.DOUBLE],dest=0,tag=1)
            if rank == 0 :
                bar = ProgressBar(number_rank,"Reconstruction of scattering absorptions will begin")
                k_sca_rmd_tot = k_sca_rmd
                for i_n in range(1,number_rank) :
                    sh_k = np.zeros(2,dtype=np.int)
                    comm.Recv([sh_k,MPI.INT],source=i_n,tag=0)
                    k_sca_rmd_n = np.zeros((sh_k),dtype=np.float64)
                    comm.Recv([k_sca_rmd_n,MPI.DOUBLE],source=i_n,tag=1)
                    k_sca_rmd_tot = np.concatenate((k_sca_rmd_tot,k_sca_rmd_n))
                    bar.animate(i_n+1)

                if Kcorr == True :
                    np.save("%s%s/k_sca_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_sca_rmd_tot)
                else :
                    np.save("%s%s/k_sca_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_sca_rmd_tot)
                del k_sca_rmd_tot
        else :
            if Kcorr == True :
                np.save("%s%s/k_sca_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain),k_sca_rmd)
            else :
                np.save("%s%s/k_sca_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain),k_sca_rmd)

        if rank == 0 :
            print "Rayleigh_scattering finished with success"

        del k_sca_rmd,x_mol_species

    else :

        if rank == rank_ref :
            print "There is no scattering"

    if rank_max == comm.size :
        comm.Barrier()

    if Clouds == True :

        zer_n, = np.where(bande_sample != 0.)
        zer_p, = np.where(bande_sample == 0.)

        if Kcorr == True :
            wl = np.zeros(bande_sample.size-1)
            for i in range(bande_sample.size - 1) :
                wl[i] = (bande_sample[i+1] + bande_sample[i])/2.
            wl = 1./wl
        else :
            wl = np.zeros(bande_sample.size)
            wl[zer_n] = 1./(100*bande_sample[zer_n])
            wl[zer_p] = 0.

        n_size,rmd_size = np.shape(composit_rmd)
        c_number = c_species.size

        Qext = np.load(Qext)

        if rank_max == comm.size :
            comm.Barrier()

        if rank == 0 :
            bar = ProgressBar(c_number*number_rank,"Reconstruction of clouds scattering absorptions will begin")

        for c_num in range(c_number) :

            k_cloud_rmd = cloud_scattering(Qext[c_num,:,:],bande_cloud,P_rmd,T_rmd,wl,composit_rmd[n_size-1,:],rho_p[c_num],gen_cond_rmd[c_num,:],r_eff[c_num],r_cloud,zero,rank,rank_ref,Kcorr)

            if ByLay == False :
                if rank != 0 :
                    sh_k = np.array(np.shape(k_cloud_rmd),dtype=np.int)
                    comm.Send([sh_k,MPI.INT],dest=0,tag=0)
                    comm.Send([k_cloud_rmd,MPI.DOUBLE],dest=0,tag=1)
                if rank == 0 :
                    k_cloud_rmd_tot = k_cloud_rmd
                    for i_n in range(1,number_rank) :
                        sh_k = np.zeros(2,dtype=np.int)
                        comm.Recv([sh_k,MPI.INT],source=i_n,tag=0)
                        k_cloud_rmd_n = np.zeros((sh_k),dtype=np.float64)
                        comm.Recv([k_cloud_rmd_n,MPI.DOUBLE],source=i_n,tag=1)
                        k_cloud_rmd_tot = np.concatenate((k_cloud_rmd_tot,k_cloud_rmd_n))
                        bar.animate(c_num*number_rank+i_n+1)

                if rank == 0 :
                    if c_num == 0 :
                        sh_c = np.shape(k_cloud_rmd_tot)
                        k_cloud_rmd_fin = np.zeros((c_number,sh_c[0],sh_c[1]),dtype=np.float64)
                        k_cloud_rmd_fin[c_num,:,:] = k_cloud_rmd_tot
                        del k_cloud_rmd_tot
                    else :
                        k_cloud_rmd_fin[c_num,:,:] = k_cloud_rmd_tot
                        del k_cloud_rmd_tot
            else :
                if c_num == 0 :
                    sh_c = np.shape(k_cloud_rmd)
                    k_cloud_rmd_fin = np.zeros((c_number,sh_c[0],sh_c[1]),dtype=np.float64)
                    k_cloud_rmd_fin[c_num,:,:] = k_cloud_rmd
                    del k_cloud_rmd
                else :
                    k_cloud_rmd_fin[c_num,:,:] = k_cloud_rmd
        r_enn = ''
        for i_r in range(r_eff.size) :
            if i_r != r_eff.size-1 :
                r_enn += '%.2f_'%(r_eff[i_r]*10**6)
            else :
                r_enn += '%.2f'%(r_eff[i_r]*10**6)
        if ByLay == False :
            if rank == 0 :
                if Kcorr == True :
                    np.save("%s%s/k_cloud_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s_%s.npy" \
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,r_enn,domain),k_cloud_rmd_fin)
                else :
                    np.save("%s%s/k_cloud_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%s.npy" \
                    %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,r_enn,domain),k_cloud_rmd_fin)
                del k_cloud_rmd_fin
        else :
            if Kcorr == True :
                np.save("%s%s/k_cloud_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s_%s.npy" \
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,r_enn,domain),k_cloud_rmd_fin)
            else :
                np.save("%s%s/k_cloud_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s_%s.npy" \
                %(directory,name,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,r_enn,domain),k_cloud_rmd_fin)
            del k_cloud_rmd_fin

        if rank == 0 :
            print "Cloud scattering finished with success, process are beginning to save data remind"

        del Qext

    else :

        if rank == 0 :
            print "There is no clouds"

    if rank_max == comm.size :
        comm.Barrier()


########################################################################################################################


def convertator1D (P_col,T_col,gen_col,c_species,Q_col,compo_col,ind_active,K,K_cont,Qext,P_sample,\
                   T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,name,t,phi_rot,\
                   n_species,domain,ratio,directory,Tracer=False,Continuum=False,Scattering=False,Clouds=False,\
                   Kcorr=True,Optimal=False,Script=True) :

    c_number = c_species.size

    if Kcorr == True :
        dim_P,dim_T,dim_x,dim_bande,dim_gauss = np.shape(K)
    else :
        n_spe,dim_P,dim_T,dim_bande = np.shape(K)
        dim_gauss = 0

    no_zero, = np.where(P_col != 0)
    T_rmd = T_col[no_zero]
    P_rmd = P_col[no_zero]
    compo_rmd = compo_col[:,no_zero]
    zero = np.array([])

    if Tracer == False :

        if Kcorr == True :
            k_rmd = Ksearcher(T_rmd,P_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,Kcorr,Optimal,Script)
            if Script == True :
                print "Ksearcher finished with success"
        else :
            compo = compo_col[ind_active,:]
            result = Ssearcher(T_rmd,P_rmd,compo,K,P_sample,T_sample,Kcorr,Optimal,Script)
            k_rmd = result
            if Script == True :
                print "Ssearcher finished with success"
        Q_rmd = np.array([])

    else :

        Q_rmd = Q_col[no_zero]

        if Kcorr == True :
            result = Ksearcher_M(T_rmd,P_rmd,Q_rmd,dim_gauss,dim_bande,K,P_sample,T_sample,Q_sample,Kcorr,Optimal,Script)
            k_rmd = result
            if Script == True :
                print "Ksearcher_M finished with success"
        else :
            compo = compo_col[ind_active,:]
            result = Ssearcher_M(T_rmd,P_rmd,Q_rmd,compo,K,P_sample,T_sample,Q_sample,Kcorr,Optimal,Script)
            k_rmd = result
            if Script == True :
                print "Ssearcher_M finished with success"

    if Continuum == True :

        cont_species = K_cont.species
        H2, He, Other = H2HeO(cont_species)

        decont = 0
        amagat = 2.69578e-3*P_rmd/T_rmd
        k_cont_rmd = np.zeros((dim_bande,P_rmd.size))

        if H2 == True :

            decont += 1
            K_cont_h2h2 = np.load('%sSource/k_cont_%s.npy'%(directory,K_cont.associations[0]))
            K_cont_nu_h2h2 = np.load('%sSource/k_cont_nu_%s.npy'%(directory,K_cont.associations[0]))
            T_cont_h2h2 = np.load('%sSource/T_cont_%s.npy'%(directory,K_cont.associations[0]))

            k_interp_h2h2 = k_cont_interp_h2h2_integration(K_cont_h2h2,K_cont_nu_h2h2,\
                                        T_rmd,bande_sample,T_cont_h2h2,1,1,Kcorr)

            amagat_h2h2 = amagat*compo_rmd[0,:]

            for i_bande in range(dim_bande) :

                k_cont_rmd[i_bande,:] = amagat_h2h2**2*k_interp_h2h2[i_bande,:]

            del amagat_h2h2,k_interp_h2h2

        if He == True :

            decont += 1
            K_cont_h2he = np.load('%sSource/k_cont_%s.npy'%(directory,K_cont.associations[1]))
            K_cont_nu_h2he = np.load('%sSource/k_cont_nu_%s.npy'%(directory,K_cont.associations[1]))
            T_cont_h2he = np.load('%sSource/T_cont_%s.npy'%(directory,K_cont.associations[1]))

            k_interp_h2he = k_cont_interp_h2he_integration(K_cont_h2he,K_cont_nu_h2he,\
                                    T_rmd,bande_sample,T_cont_h2he,1,1,Kcorr)

            amagat_self = amagat*compo_rmd[0,:]
            amagat_foreign = amagat*compo_rmd[1,:]

            for i_bande in range(dim_bande) :

                k_cont_rmd[i_bande,:] += amagat_foreign*amagat_self*k_interp_h2he[i_bande,:]

            del amagat_foreign,amagat_self,k_interp_h2he

        if Other == True :

            for i_cont in range(decont,cont_species.size) :

                K_cont_spespe = np.load('%sSource/k_cont_%s.npy'%(directory,K_cont.associations[i_cont]))
                K_cont_nu_spespe = np.load('%sSource/k_cont_nu_%s.npy'%(directory,K_cont.associations[i_cont]))
                T_cont_spespe = np.load('%sSource/T_cont_%s.npy'%(directory,K_cont.associations[i_cont]))

                if cont_species[i_cont] != 'H2O' and cont_species[i_cont] != 'H2Os':
                    wh_c, = np.where(n_species == cont_species[i_cont])
                    amagat_spefor = amagat*compo_rmd[0,:]
                    amagat_speself = amagat*compo_rmd[wh_c[0],:]
                    amagat_spe = amagat_spefor*amagat_speself
                else :
                    wh_c, = np.where(n_species == 'H2O')
                    H2O = True
                    N_mol = P_rmd/(k_B*T_rmd)
                    if cont_species[i_cont] == 'H2O' :
                        amagat_spe = amagat*(1.-compo_rmd[wh_c[0],:])*compo_rmd[wh_c[0],:]*N_mol
                    if cont_species[i_cont] == 'H2Os' :
                        amagat_spe = amagat*compo_rmd[wh_c[0],:]**2*N_mol

                k_interp_spespe = k_cont_interp_spespe_integration(K_cont_spespe,K_cont_nu_spespe,\
                            T_rmd,bande_sample,T_cont_spespe,1,1,K_cont.associations[i_cont],Kcorr,H2O)

                for i_bande in range(dim_bande) :

                    k_cont_rmd[i_bande,:] += amagat_spe*k_interp_spespe[i_bande,:]

                del amagat_spe,k_interp_spespe

        k_cont_rmd = np.transpose(k_cont_rmd)

    else :

        if Script == True :
            print 'There is no continuum'

        k_cont_rmd = np.zeros((T_rmd.size,dim_bande))

    order,len = np.shape(compo_rmd)
    x_mol_species = compo_rmd[0:order-1,:]

    if Scattering == True :

        k_sca_rmd = Rayleigh_scattering(P_rmd,T_rmd,bande_sample,x_mol_species,n_species,zero,Kcorr,True,Script)

        if Script == True :
            print "Rayleigh_scattering finished with success"

    else :

        k_sca_rmd = np.zeros((T_rmd.size,dim_bande))

    if Clouds == True :

        gen_rmd = gen_col[:,no_zero]
        zer_n, = np.where(bande_sample != 0.)
        zer_p, = np.where(bande_sample == 0.)

        if Kcorr == True :
            wl = np.zeros(bande_sample.size-1)
            for i in range(bande_sample.size - 1) :
                wl[i] = (bande_sample[i+1] + bande_sample[i])/2.
        else :
            wl = np.zeros(bande_sample.size)
            wl[zer_n] = 1./(100*bande_sample[zer_n])
            wl[zer_p] = 0.

        n_size,rmd_size = np.shape(compo_rmd)

        k_cloud_rmd = np.zeros((c_number,P_rmd.size,dim_bande))

        for c_num in range(c_number) :

            k_cloud_rmd[c_num,:,:] = cloud_scattering(Qext[c_num,:,:],bande_cloud,P_rmd,T_rmd,wl,compo_rmd[n_size-1,:],rho_p[c_num],gen_rmd[c_num,:],r_eff[c_num],r_cloud,zero,Kcorr,Script)

        if Script == True :
            print "Cloud_scattering finished with success, process are beginning to save data remind \n"

    else :

        k_cloud_rmd = np.zeros((T_rmd.size,dim_bande))


    return P_rmd,T_rmd,Q_rmd,k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd


########################################################################################################################
########################################################################################################################

"""
    K_CORRELATED_INTERP, K_CORRELATED_INTERP_BOUCLE

    La fonction exploite les profils en temperature, pression et fraction molaire pour fournir en chaque point une valeur
    d'opacite qui depend de la temperature locale, de la pression locale, de l'abondance relative locale ainsi que de la
    bande et du point de Gauss considere. Elle effectue une interpolation lineaire par rapport a la temperature et la
    fraction molaire, ainsi qu'une interpolation lineaire par rapport au logarithme de la pression.

    En somme, elle cherhe les 8 points dans l'espace T,P,x pour une bande et un point de gauss donne, retrouve les
    opacites correspondante et en deduit une valeur locale par interpolation. Cette fonction s'applique sur des tableaux
    de donnees. Une premiere iteration permet de generer un ensemble de coefficients d'interpolation et les indices
    correspondants, il est ensuite reintroduit dans la fonction _BOUCLE pour eviter d'en reiterer les calculs.

"""

########################################################################################################################
########################################################################################################################


def Ssearcher(T_array,P_array,compo_array,sigma_array,P_sample,T_sample,rank,rank_ref,Kcorr,oneD=False,Optimal=False) :

    n_sp, P_size, T_size, dim_bande = np.shape(sigma_array)

    sigma_data = sigma_array[0,:,:,:]

    sigma_array_t = np.transpose(sigma_array,(1,2,3,0))
    compo_array_t = np.transpose(compo_array)

    k_inter,size,i_Tu_arr,i_pu_arr,coeff_1_array,coeff_3_array = \
    k_correlated_interp(sigma_data,P_array,T_array,0,P_sample,T_sample,rank,rank_ref,Kcorr,oneD,Optimal)

    zz, = np.where(i_Tu_arr == 0)
    i_Td_arr = i_Tu_arr - 1
    i_Td_arr[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_arr == 0)
    i_pd_arr = i_pu_arr - 1
    i_pd_arr[zz] = np.zeros(zz.size)

    k_rmd = np.zeros((i_Tu_arr.size,dim_bande))

    if rank == rank_ref :
        bar = ProgressBar(i_Tu_arr.size,'Ssearcher progression')

    for i in xrange( i_Tu_arr.size ):

        i_Tu = i_Tu_arr[i]
        i_Td = i_Td_arr[i]
        i_pu = i_pu_arr[i]
        i_pd = i_pd_arr[i]

        if Optimal == False :

            coeff_1 = coeff_1_array[i]
            coeff_2 = 1. - coeff_1
            coeff_3 = coeff_3_array[i]
            coeff_4 = 1. - coeff_3

            c13 = coeff_1 * coeff_3 * 0.0001
            c23 = coeff_2 * coeff_3 * 0.0001
            c14 = coeff_1 * coeff_4 * 0.0001
            c24 = coeff_2 * coeff_4 * 0.0001

            k_pd_Td = sigma_array_t[i_pd, i_Td, :, :]
            k_pd_Tu = sigma_array_t[i_pd, i_Tu, :, :]
            k_pu_Td = sigma_array_t[i_pu, i_Td, :, :]
            k_pu_Tu = sigma_array_t[i_pu, i_Tu, :, :]

            comp = compo_array_t[i,:]

            k_1 = k_pd_Td * c24 + k_pd_Tu * c23
            k_2 = k_pu_Td * c14 + k_pu_Tu * c13

            k_rmd[i, :] = np.dot( k_1 + k_2, comp )

            #if rank == rank_ref :
            #    print k_pd_Td[300],k_pu_Td[300],k_pd_Tu[300],k_pu_Tu[300],k_rmd[i,300],coeff_1, coeff_2, coeff_3, coeff_4, c13,c23,c14,c24

        else :

            b_m = coeff_1_array[0,i]
            a_m = coeff_1_array[1,i]
            T = coeff_1_array[2,i]
            coeff_3 = coeff_3_array[i] * 0.0001
            coeff_4 = (1. - coeff_3) * 0.0001

            k_pd_Tu = sigma_array_t[i_pd, i_Tu, :, :]
            k_pu_Tu = sigma_array_t[i_pu, i_Tu, :, :]

            comp = compo_array_t[i,:]

            k_u = k_pd_Tu * coeff_4 + k_pu_Tu * coeff_3
            k_d = k_pd_Td * coeff_4 + k_pu_Td * coeff_3

            b_mm = b_m * np.log(k_u/k_d)
            a_mm = np.exp(b_mm/a_m)

            k_rmd[i, :] = np.dot( k_u*a_mm*np.exp(-b_mm/T), comp )

        if rank == rank_ref :
            bar.animate( i+1 )

    return k_rmd


########################################################################################################################


def Ksearcher(T_array,P_array,dim_gauss,dim_bande,k_corr_data_grid,P_sample,T_sample,rank,rank_ref,Kcorr,oneD=False,Optimal=False) :

    k_rmd = np.zeros((P_array.size,dim_bande,dim_gauss))

    layer = int(T_array.size/10.)

    T_size = T_array.size

    if oneD == False :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande*dim_gauss,'K-correlated coefficients computation')

    for i_bande in range(dim_bande) :

        k_corr_data = k_corr_data_grid[:,:,:,i_bande,:]

        if i_bande == 0 :

            for i_gauss in range(dim_gauss) :

                if i_gauss == 0 :
                    k_inter,size,i_Tu_array,i_pu_array,coeff_1_array,coeff_3_array = \
                    k_correlated_interp(k_corr_data,P_array,T_array,i_gauss,P_sample,T_sample,rank,rank_ref,Kcorr,oneD,Optimal)

                    k_rmd[:,i_bande,i_gauss] = k_inter[:]

                else :

                    for lay in range(10) :

                        if lay != 9 :

                            k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],\
                                            coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],i_gauss,Optimal,Kcorr)

                            k_rmd[lay*layer:(lay+1)*layer,i_bande,i_gauss] = k_inter[:]

                        else :

                            k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],\
                                            coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],i_gauss,Optimal,Kcorr)

                            k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if oneD == False :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

        else :

            for i_gauss in range(dim_gauss) :

                for lay in range(10) :

                    if lay != 9 :

                        k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],\
                                            coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],i_gauss,Optimal,Kcorr)

                        k_rmd[lay*layer:(lay+1)*layer,i_bande,i_gauss] = k_inter[:]

                    else :

                        k_inter = k_correlated_interp_boucle(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],\
                                            coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],i_gauss,Optimal,Kcorr)

                        k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if oneD == False :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

    #print("Computed for bande %i" %(i_bande+1))

    return k_rmd


########################################################################################################################


def k_correlated_interp(k_corr_data,P_array,T_array,i_gauss,P_sample,T_sample,rank,rank_ref,Kcorr=True,oneD=False,Optimal=False) :

    size = P_array.size
    k_inter = np.zeros(size)

    i_Tu_array = np.zeros(size, dtype = "int")
    i_pu_array = np.zeros(size, dtype = "int")
    if Optimal == True :
        coeff_1_array = np.zeros((3,size))
    else :
        coeff_1_array = np.zeros(size)
    coeff_3_array = np.zeros(size)

    if rank == rank_ref :
        bar = ProgressBar(size,'Module interpolates cross sections')

    for i in range(size) :

        P = np.log10(P_array[i])-2
        T = T_array[i]

        if T == 0 or P == 0 :
            i_Tu = 0
            i_pu = 0
            coeff_1 = 0
            coeff_3 = 0
            k_inter[i] = 0.

        else :

            if T == T_array[i-1] and P == P_array[i-1] and i != 0 :
                k_inter[i] = k_inter[i-1]
            else :
                if Kcorr == True :
                    res,c_grid,i_grid = interp2olation_opti_uni(T,P,T_sample,P_sample,k_corr_data[:,:,0,i_gauss],False,False)
                    k_inter[i] = res
                    i_Td, i_Tu, i_pd, i_pu = i_grid[0], i_grid[1], i_grid[2], i_grid[3]
                    coeff_3, coeff_1 = c_grid[0], c_grid[2]
                else :
                    if Optimal == True :
                        res,c_grid,i_grid = interp2olation_opti_uni(P,T,P_sample,T_sample,k_corr_data[:,:,0],False,True)
                        b_m, a_m, T = c_grid[0], c_grid[1], c_grid[2]
                        coeff_3 = c_grid[3]
                        k_inter[i] = res
                    else :
                        res,c_grid,i_grid = interp2olation_opti_uni(P,T,P_sample,T_sample,k_corr_data[:,:,0],False,False)
                        coeff_1, coeff_3 = c_grid[0], c_grid[2]
                        k_inter[i] = res
                    i_pd, i_pu, i_Td, i_Tu = i_grid[0], i_grid[1], i_grid[2], i_grid[3]

        i_Tu_array[i] = int(i_Tu)
        i_pu_array[i] = int(i_pu)
        if Optimal == False :
            coeff_1_array[i] = coeff_1
        else :
            coeff_1_array[0,i] = b_m
            coeff_1_array[1,i] = a_m
            coeff_1_array[2,i] = T
        coeff_3_array[i] = coeff_3

        if rank == rank_ref :
            if i%100 == 0. or i == size - 1 :
                bar.animate(i + 1)
            #print T, P, i_Tu, i_Td, T_sample[i_Tu], T_sample[i_Td], i_pu, i_pd, P_sample[i_pu], P_sample[i_pd]

    return k_inter*0.0001,size,i_Tu_array,i_pu_array,coeff_1_array,coeff_3_array

########################################################################################################################

def k_correlated_interp_boucle(k_corr_data,size,i_Tu_array,i_pu_array,\
                               coeff_1_array,coeff_3_array,i_gauss,Optimal=False,Kcorr=False):

    if Optimal == False :
        coeff_2_array = 1 - coeff_1_array

    coeff_4_array = 1 - coeff_3_array
    zz, = np.where(i_Tu_array == 0)
    i_Td_array = i_Tu_array - 1
    i_Td_array[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_array == 0)
    i_pd_array = i_pu_array - 1
    i_pd_array[zz] = np.zeros(zz.size)

    gauss = np.ones(i_Tu_array.size,dtype='int')*i_gauss

    if Kcorr == True :
        k_pd_Td = k_corr_data[i_Td_array,i_pd_array,0,gauss]
        k_pd_Tu = k_corr_data[i_Tu_array,i_pd_array,0,gauss]
        k_pu_Td = k_corr_data[i_Td_array,i_pu_array,0,gauss]
        k_pu_Tu = k_corr_data[i_Tu_array,i_pu_array,0,gauss]
    else :
        k_pd_Td = k_corr_data[i_pd_array,i_Td_array,0]
        k_pd_Tu = k_corr_data[i_pd_array,i_Tu_array,0]
        k_pu_Td = k_corr_data[i_pu_array,i_Td_array,0]
        k_pu_Tu = k_corr_data[i_pu_array,i_Tu_array,0]

    k_d = k_pd_Td*coeff_4_array + k_pu_Td*coeff_3_array
    k_u = k_pd_Tu*coeff_4_array + k_pu_Tu*coeff_3_array
    if Optimal == False :
        k_inter = k_d*coeff_2_array + k_u*coeff_1_array
    else :
        b_m = coeff_1_array[0]
        a_m = coeff_1_array[1]
        T = coeff_1_array[2]
        wh_zn, = np.where(k_d != 0.00)
        b_mm = np.zeros(b_m.size,dtype=np.float64)
        b_mm[wh_zn] = b_m*np.log(k_u[wh_zn]/k_d[wh_zn])
        a_mm = np.exp(b_mm/a_m)
        k_inter = k_u*a_mm*np.exp(b_mm/T)

    return k_inter*0.0001


########################################################################################################################


def Ksearcher_M(T_array,P_array,Q_array,dim_gauss,dim_bande,k_corr_data_grid,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,oneD=False,Optimal=False) :

    k_rmd = np.zeros((P_array.size,dim_bande,dim_gauss))

    if oneD == False :
        if rank == rank_ref :
            bar = ProgressBar(dim_bande*dim_gauss,'K-correlated coefficients computation')

    layer = int(T_array.size/10.)

    T_size = T_array.size

    for i_bande in range(dim_bande) :

        k_corr_data = k_corr_data_grid[:,:,:,i_bande,:]

        if i_bande == 0 :

            for i_gauss in range(dim_gauss) :

                if i_gauss == 0 :

                    k_inter,size,i_Tu_array,i_pu_array,i_Qu_array,coeff_1_array,coeff_3_array,coeff_5_array = \
                        k_correlated_interp_M(k_corr_data,P_array,T_array,Q_array,i_gauss,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,oneD,Optimal)

                    k_rmd[:,i_bande,i_gauss] = k_inter[:]

                else :

                    for lay in range(5) :

                        if lay != 4 :

                            k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],i_Qu_array[lay*layer:(lay+1)*layer],\
                                                    coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],coeff_5_array[lay*layer:(lay+1)*layer],i_gauss,Optimal,Kcorr)


                            k_rmd[lay*layer:(lay+1)*layer,i_gauss,i_bande] = k_inter[:]

                        else :

                            k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],i_Qu_array[lay*layer:T_size],\
                                                    coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],coeff_5_array[lay*layer:T_size],i_gauss,Optimal,Kcorr)

                            k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]


                if oneD == False :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

        else :

            for i_gauss in range(dim_gauss) :

                for lay in range(5) :

                    if lay != 4 :

                        k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:(lay+1)*layer],i_pu_array[lay*layer:(lay+1)*layer],i_Qu_array[lay*layer:(lay+1)*layer],\
                                                    coeff_1_array[lay*layer:(lay+1)*layer],coeff_3_array[lay*layer:(lay+1)*layer],coeff_5_array[lay*layer:(lay+1)*layer],i_gauss,Optimal,Kcorr)


                        k_rmd[lay*layer:(lay+1)*layer,i_bande,i_gauss] = k_inter[:]

                    else :

                        k_inter = k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array[lay*layer:T_size],i_pu_array[lay*layer:T_size],i_Qu_array[lay*layer:T_size],\
                                                    coeff_1_array[lay*layer:T_size],coeff_3_array[lay*layer:T_size],coeff_5_array[lay*layer:T_size],i_gauss,Optimal,Kcorr)

                        k_rmd[lay*layer:T_size,i_bande,i_gauss] = k_inter[:]

                if oneD == False :
                    if rank == rank_ref :
                        bar.animate(i_bande*dim_gauss + i_gauss + 1)

    return k_rmd


########################################################################################################################


def Ssearcher_M(T_array,P_array,Q_array,compo_array,sigma_array,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,oneD=False,Optimal=False) :

    n_sp, P_size, T_size, dim_bande = np.shape(sigma_array)

    sigma_data = sigma_array[0,:,:,:]

    sigma_array_t = np.transpose(sigma_array,(1,2,3,4,0))
    compo_array_t = np.transpose(compo_array)

    k_inter,size,i_Tu_arr,i_pu_arr,i_Qu_arr,coeff_1_array,coeff_3_array,coeff_5_array = \
    k_correlated_interp_M(sigma_data,P_array,T_array,Q_array,0,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr,Optimal)

    zz, = np.where(i_Tu_arr == 0)
    i_Td_arr = i_Tu_arr - 1
    i_Td_arr[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_arr == 0)
    i_pd_arr = i_pu_arr - 1
    i_pd_arr[zz] = np.zeros(zz.size)
    zz, = np.where(i_Qu_arr == 0)
    i_Qd_arr = i_Qu_arr - 1
    i_Qd_arr[zz] = np.zeros(zz.size)

    k_rmd = np.zeros((i_Tu_arr.size,dim_bande))

    if rank == rank_ref :
        bar = ProgressBar(i_Tu_arr.size,'Ssearcher progression')

    for i in xrange( i_Tu_arr.size ):

        i_Tu = i_Tu_arr[i]
        i_Td = i_Td_arr[i]
        i_pu = i_pu_arr[i]
        i_pd = i_pd_arr[i]
        i_Qu = i_Qu_arr[i]
        i_Qd = i_Qd_arr[i]

        if Optimal == False :

            coeff_1 = coeff_1_array[i]
            coeff_2 = 1. - coeff_1
            coeff_3 = coeff_3_array[i]
            coeff_4 = 1. - coeff_3
            coeff_5 = coeff_5_array[i]
            coeff_6 = 1. - coeff_5

            c425 = coeff_2 * coeff_5 * coeff_4 * 0.0001
            c415 = coeff_1 * coeff_5 * coeff_4 * 0.0001
            c325 = coeff_2 * coeff_5 * coeff_3 * 0.0001
            c315 = coeff_1 * coeff_5 * coeff_3 * 0.0001
            c426 = coeff_2 * coeff_6 * coeff_4 * 0.0001
            c416 = coeff_1 * coeff_6 * coeff_4 * 0.0001
            c326 = coeff_2 * coeff_6 * coeff_3 * 0.0001
            c316 = coeff_1 * coeff_6 * coeff_3 * 0.0001

            k_pd_Td_Qu = sigma_array_t[i_pd, i_Td, i_Qu, :, :]
            k_pd_Tu_Qu = sigma_array_t[i_pd, i_Tu, i_Qu, :, :]
            k_pu_Td_Qu = sigma_array_t[i_pu, i_Td, i_Qu, :, :]
            k_pu_Tu_Qu = sigma_array_t[i_pu, i_Tu, i_Qu, :, :]
            k_pd_Td_Qd = sigma_array_t[i_pd, i_Td, i_Qd, :, :]
            k_pd_Tu_Qd = sigma_array_t[i_pd, i_Tu, i_Qd, :, :]
            k_pu_Td_Qd = sigma_array_t[i_pu, i_Td, i_Qd, :, :]
            k_pu_Tu_Qd = sigma_array_t[i_pu, i_Tu, i_Qd, :, :]

            comp = compo_array_t[i,:]

            k_1 = k_pd_Td_Qu * c425 + k_pd_Tu_Qu * c325
            k_2 = k_pu_Td_Qu * c415 + k_pu_Tu_Qu * c315
            k_3 = k_pd_Td_Qd * c426 + k_pd_Tu_Qd * c326
            k_4 = k_pu_Td_Qd * c416 + k_pu_Tu_Qd * c316

            k_rmd[i, :] = np.dot(k_1 + k_2 + k_3 + k_4, comp )

        else :

            b_m = coeff_1_array[0,i]
            a_m = coeff_1_array[1,i]
            T = coeff_1_array[2,i]
            coeff_3 = coeff_3_array[i]
            coeff_4 = 1. - coeff_3
            coeff_5 = coeff_5_array[i]
            coeff_6 = 1. - coeff_5

            c45 = coeff_5 * coeff_4 * 0.0001
            c35 = coeff_5 * coeff_3 * 0.0001
            c46 = coeff_6 * coeff_4 * 0.0001
            c36 = coeff_6 * coeff_3 * 0.0001

            k_pd_Tu_Qu = sigma_array_t[i_pd, i_Tu, i_Qu, :, :]
            k_pu_Tu_Qu = sigma_array_t[i_pu, i_Tu, i_Qu, :, :]
            k_pd_Tu_Qd = sigma_array_t[i_pd, i_Tu, i_Qd, :, :]
            k_pu_Tu_Qd = sigma_array_t[i_pu, i_Tu, i_Qd, :, :]

            comp = compo_array_t[i,:]

            k_1 = k_pd_Tu_Qd * c46 + k_pd_Tu_Qu * c45
            k_2 = k_pu_Tu_Qd * c36 + k_pu_Tu_Qu * c35
            k_3 = k_pd_Td_Qd * c46 + k_pd_Td_Qu * c45
            k_4 = k_pu_Td_Qd * c36 + k_pu_Td_Qu * c35

            b_mm = b_m * np.log((k_1+k_2)/(k_3+k_4))
            a_mm = np.exp(b_mm/a_m)

            k_rmd[i, :] = np.dot((k_1+k_2)*a_mm*np.exp(-b_mm/T), comp )

        if rank == rank_ref :
            bar.animate( i+1 )

    return k_rmd


########################################################################################################################


def k_correlated_interp_M(k_corr_data,P_array,T_array,Q_array,i_gauss,P_sample,T_sample,Q_sample,rank,rank_ref,Kcorr=True,oneD=False,Optimal=False) :

    T_min = T_sample[0]
    p_min = P_sample[0]
    q_min = np.log10(Q_sample[0])

    size = P_array.size
    k_inter = np.zeros(size)
    i_pu_ref = 100
    i_Tu_ref = 100
    i_Qu_ref = 100

    if Kcorr == True :
        k_lim_ddd = k_corr_data[0,0,0,i_gauss]
    else :
        k_lim_ddd = k_corr_data[0,0,0]

    i_Tu_array = np.zeros(size, dtype = "int")
    i_pu_array = np.zeros(size, dtype = "int")
    i_Qu_array = np.zeros(size, dtype = "int")
    if Optimal == False :
        coeff_1_array = np.zeros(size)
    else :
        coeff_1_array = np.zeros((3,size))
    coeff_3_array = np.zeros(size)
    coeff_5_array = np.zeros(size)

    if oneD == False :
        if rank == rank_ref :
            bar = ProgressBar(size,'Module interpolates cross sections')

    for i in range(size) :

        P = P_array[i]
        T = T_array[i]
        Q = Q_array[i]
        q = np.log10(Q)
        p = np.log10(P)

        if T == 0 or P == 0 or Q == 0 :

            i_Tu = 0
            i_pu = 0
            coeff_1 = 0
            coeff_3 = 0
            k_inter[i] = 0.

        else :

            if p - 2 < p_min or T < T_min or q < q_min :

                if T == T_array[i-1] and P == P_array[i-1] and Q == Q_array[i-1] and i != 0 :

                    k_inter[i] = k_inter[i-1]

                else :

                    if p - 2 < p_min :

                        i_pu = 0
                        coeff_3 = 0

                        if T < T_min :

                            i_Tu = 0
                            coeff_1 = 0

                            if q < q_min :

                                i_Qu = 0
                                coeff_5 = 0

                                k_inter[i] = k_lim_ddd

                            else :

                                wh, = np.where((Q_sample >= Q))

                                if wh.size != 0 :

                                    i_Qu = wh[0]
                                    qu = np.log10(Q_sample[i_Qu])
                                    i_Qd = wh[0] - 1
                                    qd = np.log10(Q_sample[i_Qd])

                                    coeff_5 = (q - qd)/float(qu - qd)
                                    coeff_6 = 1 - coeff_5

                                    if Kcorr == True :
                                        k_1 = k_corr_data[0,0,i_Qd,i_gauss]
                                        k_2 = k_corr_data[0,0,i_Qu,i_gauss]

                                    else :
                                        k_1 = k_corr_data[0,0,i_Qd]
                                        k_2 = k_corr_data[0,0,i_Qu]

                                    k_inter[i] = k_1*coeff_6 + k_2*coeff_5

                                else :
                                    i_Qu = Q_sample.size - 1
                                    i_Qd = Q_sample.size - 1

                                    coeff_5 = 1.
                                    coeff_6 = 0.

                                    if Kcorr == True :
                                        k_inter[i] = k_corr_data[0,0,i_Qu,i_gauss]

                                    else :
                                        k_inter[i] = k_corr_data[0,0,i_Qu]


                        else :

                            wh, = np.where((T_sample >= T))

                            if wh.size != 0 :

                                i_Tu = wh[0]
                                Tu = T_sample[i_Tu]
                                i_Td = wh[0] - 1
                                Td = T_sample[i_Td]

                                coeff_1 = (T - Td)/float(Tu - Td)
                                coeff_2 = 1 - coeff_1

                                if q < q_min :

                                    i_Qu = 0
                                    coeff_5 = 0

                                    if Kcorr == True :
                                        k_1 = k_corr_data[i_Td,0,0,i_gauss]
                                        k_2 = k_corr_data[i_Tu,0,0,i_gauss]

                                    else :
                                        k_1 = k_corr_data[0,i_Td,0]
                                        k_2 = k_corr_data[0,i_Tu,0]

                                    if Optimal == False :
                                        k_inter[i] = k_1*coeff_2 + k_2*coeff_1
                                    else :
                                        b_m = (Tu - Td)/(Td*Tu)*np.log(k_2/k_1)
                                        a_m = np.exp(b_m/Tu)
                                        k_inter[i] = k_2*a_m*np.exp(b_m/T)

                                else :

                                    wh, = np.where((Q_sample >= Q))

                                    if wh.size != 0 :

                                        i_Qu = wh[0]
                                        qu = np.log10(Q_sample[i_Qu])
                                        i_Qd = wh[0] - 1
                                        qd = np.log10(Q_sample[i_Qd])

                                        coeff_5 = (q - qd)/float(qu - qd)
                                        coeff_6 = 1 - coeff_5

                                        if Kcorr == True :
                                            k_1 = k_corr_data[i_Td,0,i_Qd,i_gauss]
                                            k_2 = k_corr_data[i_Tu,0,i_Qd,i_gauss]
                                            k_3 = k_corr_data[i_Td,0,i_Qu,i_gauss]
                                            k_4 = k_corr_data[i_Tu,0,i_Qu,i_gauss]

                                        else :
                                            k_1 = k_corr_data[0,i_Td,i_Qd]
                                            k_2 = k_corr_data[0,i_Tu,i_Qd]
                                            k_3 = k_corr_data[0,i_Td,i_Qu]
                                            k_4 = k_corr_data[0,i_Tu,i_Qu]

                                        k_5 = k_1*coeff_6 + k_3*coeff_5
                                        k_6 = k_2*coeff_6 + k_4*coeff_5

                                        if Optimal == False :
                                            k_inter[i] = k_5*coeff_2 + k_6*coeff_1
                                        else :
                                            b_m = (Tu - Td)/(Td*Tu)*np.log(k_6/k_5)
                                            a_m = np.exp(b_m/Tu)
                                            k_inter[i] = k_6*a_m*np.exp(b_m/T)

                                    else :
                                        i_Qu = Q_sample.size - 1
                                        i_Qd = Q_sample.size - 1

                                        coeff_5 = 1.
                                        coeff_6 = 0.

                                        if Kcorr == True :
                                            k_1 = k_corr_data[i_Td,0,i_Qu,i_gauss]
                                            k_2 = k_corr_data[i_Tu,0,i_Qu,i_gauss]

                                        else :
                                            k_1 = k_corr_data[0,i_Td,i_Qu]
                                            k_2 = k_corr_data[0,i_Tu,i_Qu]

                                        if Optimal == False :
                                            k_inter[i] = k_1*coeff_2 + k_2*coeff_1
                                        else :
                                            b_m = (Tu - Td)/(Td*Tu)*np.log(k_2/k_1)
                                            a_m = np.exp(b_m/Tu)
                                            k_inter[i] = k_2*a_m*np.exp(b_m/T)

                            else :
                                i_Tu = T_sample.size - 1
                                i_Td = T_sample.size - 1

                                coeff_1 = 1.
                                coeff_2 = 0.

                                if q < q_min :

                                    i_Qu = 0
                                    coeff_5 = 0

                                    if Kcorr == True :
                                        k_inter[i] = k_corr_data[i_Tu,0,0,i_gauss]

                                    else :
                                        k_inter[i] = k_corr_data[0,i_Tu,0]

                                else :

                                    wh, = np.where((Q_sample >= Q))

                                    if wh.size != 0 :

                                        i_Qu = wh[0]
                                        qu = np.log10(Q_sample[i_Qu])
                                        i_Qd = wh[0] - 1
                                        qd = np.log10(Q_sample[i_Qd])

                                        coeff_5 = (q - qd)/float(qu - qd)
                                        coeff_6 = 1 - coeff_5

                                        if Kcorr == True :
                                            k_1 = k_corr_data[i_Tu,0,i_Qd,i_gauss]
                                            k_2 = k_corr_data[i_Tu,0,i_Qu,i_gauss]

                                        else :
                                            k_1 = k_corr_data[0,i_Tu,i_Qd]
                                            k_2 = k_corr_data[0,i_Tu,i_Qu]

                                        k_inter[i] = k_1*coeff_6 + k_2*coeff_5

                                    else :
                                        i_Qu = Q_sample.size - 1
                                        i_Qd = Q_sample.size - 1

                                        coeff_5 = 1.
                                        coeff_6 = 0.

                                        if Kcorr == True :
                                            k_inter[i] = k_corr_data[i_Tu,0,i_Qu,i_gauss]

                                        else :
                                            k_inter[i] = k_corr_data[0,i_Tu,i_Qu]

                    else :

                        p = np.log10(P) - 2

                        wh, = np.where(P_sample >= p)

                        if wh.size != 0 :
                            i_pu = wh[0]
                            pu = P_sample[i_pu]
                            i_pd = i_pu - 1
                            pd = P_sample[i_pd]

                            coeff_3 = (p - pd)/float(pu - pd)
                            coeff_4 = 1 - coeff_3

                        else :
                            i_pd = P_sample.size - 1
                            i_pu = P_sample.size - 1

                            coeff_3 = 1
                            coeff_4 = 0

                        if T < T_min :

                            i_Tu = 0

                            coeff_1 = 0

                            if q < q_min :

                                i_Qu = 0
                                coeff_5 = 0

                                if Kcorr == True :
                                    k_3 = k_corr_data[0,i_pd,0,i_gauss]
                                    k_4 = k_corr_data[0,i_pu,0,i_gauss]
                                else :
                                    k_3 = k_corr_data[i_pd,0,0]
                                    k_4 = k_corr_data[i_pu,0,0]

                                k_inter[i] = coeff_4*k_3 + coeff_3*k_4

                            else :

                                wh, = np.where((Q_sample >= Q))

                                if wh.size != 0 :

                                    i_Qu = wh[0]
                                    qu = np.log10(Q_sample[i_Qu])
                                    i_Qd = wh[0] - 1
                                    qd = np.log10(Q_sample[i_Qd])

                                    coeff_5 = (q - qd)/float(qu - qd)
                                    coeff_6 = 1 - coeff_5

                                    if Kcorr == True :
                                        k_1 = k_corr_data[0,i_pd,i_Qd,i_gauss]
                                        k_2 = k_corr_data[0,i_pu,i_Qd,i_gauss]
                                        k_3 = k_corr_data[0,i_pd,i_Qu,i_gauss]
                                        k_4 = k_corr_data[0,i_pu,i_Qu,i_gauss]
                                    else :
                                        k_1 = k_corr_data[i_pd,0,i_Qd]
                                        k_2 = k_corr_data[i_pu,0,i_Qd]
                                        k_3 = k_corr_data[i_pd,0,i_Qu]
                                        k_4 = k_corr_data[i_pu,0,i_Qu]

                                    k_5 = k_1*coeff_6 + k_3*coeff_5
                                    k_6 = k_2*coeff_6 + k_4*coeff_5

                                    k_inter[i] = coeff_4*k_5 + coeff_3*k_6

                                else :

                                    i_Qu = Q_sample.size - 1
                                    i_Qd = Q_sample.size - 1

                                    coeff_5 = 1.
                                    coeff_6 = 0.

                                    if Kcorr == True :
                                        k_1 = k_corr_data[0,i_pd,i_Qu,i_gauss]
                                        k_2 = k_corr_data[0,i_pu,i_Qu,i_gauss]

                                    else :
                                        k_1 = k_corr_data[i_pd,0,i_Qu]
                                        k_2 = k_corr_data[i_pu,0,i_Qu]

                                    k_inter[i] = coeff_4*k_1 + coeff_3*k_2

                        else :

                            wh, = np.where((T_sample >= T))

                            if wh.size != 0 :
                                i_Tu = wh[0]
                                Tu = T_sample[i_Tu]
                                i_Td = wh[0] - 1
                                Td = T_sample[i_Td]

                                coeff_1 = (T - Td)/float(Tu - Td)
                                coeff_2 = 1 - coeff_1

                            else :
                                i_Tu = T_sample.size - 1
                                i_Td = T_sample.size - 1

                                coeff_1 = 1.
                                coeff_2 = 0.

                            if Kcorr == True :
                                k_pd_Td = k_corr_data[i_Td,i_pd,0,i_gauss]
                                k_pd_Tu = k_corr_data[i_Tu,i_pd,0,i_gauss]
                                k_pu_Td = k_corr_data[i_Td,i_pu,0,i_gauss]
                                k_pu_Tu = k_corr_data[i_Tu,i_pu,0,i_gauss]
                            else :
                                k_pd_Td = k_corr_data[i_pd,i_Td,0]
                                k_pd_Tu = k_corr_data[i_pd,i_Tu,0]
                                k_pu_Td = k_corr_data[i_pu,i_Td,0]
                                k_pu_Tu = k_corr_data[i_pu,i_Tu,0]

                            k_d = k_pd_Td*coeff_4 + k_pu_Td*coeff_3
                            k_u = k_pd_Tu*coeff_4 + k_pu_Tu*coeff_3

                            if Optimal == False or wh.size == 0 :
                                k_inter[i] = k_d*coeff_2 + k_u*coeff_1
                            else :
                                b_m = (Tu - Td)/(Td*Tu)*np.log(k_u/k_d)
                                a_m = np.exp(b_m/Tu)
                                k_inter[i] = k_u*a_m*np.exp(b_m/T)

            else :


                if T == T_array[i-1] and P == P_array[i-1] and Q == Q_array[i-1] and i != 0 :

                    k_inter[i] = k_inter[i-1]

                else :

                    p = np.log10(P) - 2
                    q = np.log10(Q)
                    # le GCM donne les pressions en Pa, tandis que la table des k-correles est range avec des pressions en mbar

                    wh, = np.where(P_sample >= p)

                    if wh.size != 0 :
                        i_pu = wh[0]
                        pu = P_sample[i_pu]
                        i_pd = i_pu - 1
                        pd = P_sample[i_pd]

                        coeff_3 = (p - pd)/float(pu - pd)
                        coeff_4 = 1 - coeff_3

                    else :
                        i_pd = P_sample.size - 1
                        i_pu = P_sample.size - 1

                        coeff_3 = 1.
                        coeff_4 = 0.

                    wh, = np.where((T_sample >= T))

                    if wh.size != 0 :
                        i_Tu = wh[0]
                        Tu = T_sample[i_Tu]
                        i_Td = wh[0] - 1
                        Td = T_sample[i_Td]

                        coeff_1 = (T - Td)/float(Tu - Td)
                        coeff_2 = 1 - coeff_1
                    else :
                        i_Td = T_sample.size - 1
                        i_Tu = T_sample.size - 1

                        coeff_1 = 1.
                        coeff_2 = 0.

                    wh, = np.where((Q_sample >= Q))

                    if wh.size != 0 :
                        i_Qu = wh[0]
                        qu = np.log10(Q_sample[i_Qu])
                        i_Qd = wh[0] - 1
                        qd = np.log10(Q_sample[i_Qd])

                        coeff_5 = (q - qd)/float(qu - qd)
                        coeff_6 = 1 - coeff_5
                    else :
                        i_Qd = Q_sample.size - 1
                        i_Qu = Q_sample.size - 1

                        coeff_5 = 1.
                        coeff_6 = 0.

                    if i_pu == i_pu_ref and i_Tu == i_Tu_ref and i_Qu == i_Qu_ref :

                        k_1 = k_pd_Td_Qd*coeff_6 + k_pd_Td_Qu*coeff_5
                        k_2 = k_pu_Td_Qd*coeff_6 + k_pu_Td_Qu*coeff_5
                        k_3 = k_pd_Tu_Qd*coeff_6 + k_pd_Tu_Qu*coeff_5
                        k_4 = k_pu_Tu_Qd*coeff_6 + k_pu_Tu_Qu*coeff_5

                        k_5 = k_1*coeff_4 + k_2*coeff_3
                        k_6 = k_3*coeff_4 + k_4*coeff_3

                        if Optimal == False :
                            k_inter[i] = k_5*coeff_2 + k_6*coeff_1
                        else :
                            b_m = (Tu - Td)/(Td*Tu)*np.log(k_6/k_5)
                            a_m = np.exp(b_m/Tu)
                            k_inter[i] = k_6*a_m*np.exp(b_m/T)

                    else :

                        if Kcorr == True :
                            k_pd_Td_Qd = k_corr_data[i_Td,i_pd,i_Qd,i_gauss]
                            k_pd_Tu_Qd = k_corr_data[i_Tu,i_pd,i_Qd,i_gauss]
                            k_pu_Td_Qd = k_corr_data[i_Td,i_pu,i_Qd,i_gauss]
                            k_pu_Tu_Qd = k_corr_data[i_Tu,i_pu,i_Qd,i_gauss]
                            k_pd_Td_Qu = k_corr_data[i_Td,i_pd,i_Qu,i_gauss]
                            k_pd_Tu_Qu = k_corr_data[i_Tu,i_pd,i_Qu,i_gauss]
                            k_pu_Td_Qu = k_corr_data[i_Td,i_pu,i_Qu,i_gauss]
                            k_pu_Tu_Qu = k_corr_data[i_Tu,i_pu,i_Qu,i_gauss]
                        else :
                            k_pd_Td_Qd = k_corr_data[i_pd,i_Td,i_Qd]
                            k_pd_Tu_Qd = k_corr_data[i_pd,i_Tu,i_Qd]
                            k_pu_Td_Qd = k_corr_data[i_pu,i_Td,i_Qd]
                            k_pu_Tu_Qd = k_corr_data[i_pu,i_Tu,i_Qd]
                            k_pd_Td_Qu = k_corr_data[i_pd,i_Td,i_Qu]
                            k_pd_Tu_Qu = k_corr_data[i_pd,i_Tu,i_Qu]
                            k_pu_Td_Qu = k_corr_data[i_pu,i_Td,i_Qu]
                            k_pu_Tu_Qu = k_corr_data[i_pu,i_Tu,i_Qu]

                        k_1 = k_pd_Td_Qd*coeff_6 + k_pd_Td_Qu*coeff_5
                        k_2 = k_pu_Td_Qd*coeff_6 + k_pu_Td_Qu*coeff_5
                        k_3 = k_pd_Tu_Qd*coeff_6 + k_pd_Tu_Qu*coeff_5
                        k_4 = k_pu_Tu_Qd*coeff_6 + k_pu_Tu_Qu*coeff_5

                        k_5 = k_1*coeff_4 + k_2*coeff_3
                        k_6 = k_3*coeff_4 + k_4*coeff_3

                        if Optimal == False :
                            k_inter[i] = k_5*coeff_2 + k_6*coeff_1
                        else :
                            b_m = (Tu - Td)/(Td*Tu)*np.log(k_6/k_5)
                            a_m = np.exp(b_m/Tu)
                            k_inter[i] = k_6*a_m*np.exp(b_m/T)

                        # k_inter est un tableau de k dont les dimensions sont celles d'une section efficace, des cm^2/molecule
                        # donc en sortie nous le convertissons en m^2/molecule pour effectuer le calcul de la profondeur optique

                        i_pu_ref = i_pu
                        i_Tu_ref = i_Tu
                        i_Qu_ref = i_Qu

        i_Tu_array[i] = int(i_Tu)
        i_pu_array[i] = int(i_pu)
        i_Qu_array[i] = int(i_Qu)
        if Optimal == False :
            coeff_1_array[i] = coeff_1
        else :
            if T < T_min or T > T_sample[T_sample.size - 1] :
                coeff_1_array[1,i] = coeff_1
            else :
                coeff_1_array[0,i] = b_m
                coeff_1_array[1,i] = a_m
                coeff_1_array[2,i] = T
        coeff_3_array[i] = coeff_3
        coeff_5_array[i] = coeff_5

        if oneD == False :
            if rank == rank_ref :
                if i%100 == 0. or i == size - 1 :
                    bar.animate(i + 1)

    return k_inter*0.0001,size,i_Tu_array,i_pu_array,i_Qu_array,coeff_1_array,coeff_3_array,coeff_5_array


########################################################################################################################


def k_correlated_interp_boucle_M(k_corr_data,size,i_Tu_array,i_pu_array,i_Qu_array,\
                               coeff_1_array,coeff_3_array,coeff_5_array,i_gauss,Optimal=False,Kcorr=False):

    if Optimal == False :
        coeff_2_array = 1 - coeff_1_array
    else :
        k_inter = np.zeros(size)
        zer_n, = np.where(coeff_1_array[0,:] != 0)
        zer_p, = np.where(coeff_1_array[0,:] == 0)
    coeff_4_array = 1 - coeff_3_array
    coeff_6_array = 1 - coeff_5_array
    zz, = np.where(i_Tu_array == 0)
    i_Td_array = i_Tu_array - 1
    i_Td_array[zz] = np.zeros(zz.size)
    zz, = np.where(i_pu_array == 0)
    i_pd_array = i_pu_array - 1
    i_pd_array[zz] = np.zeros(zz.size)
    zz, = np.where(i_Qu_array == 0)
    i_Qd_array = i_Qu_array - 1
    i_Qd_array[zz] = np.zeros(zz.size)

    gauss = np.ones(i_Tu_array.size,dtype='int')*i_gauss

    if Kcorr == True :
        k_pd_Td_Qd = k_corr_data[i_Td_array,i_pd_array,i_Qd_array,gauss]
        k_pd_Tu_Qd = k_corr_data[i_Tu_array,i_pd_array,i_Qd_array,gauss]
        k_pu_Td_Qd = k_corr_data[i_Td_array,i_pu_array,i_Qd_array,gauss]
        k_pu_Tu_Qd = k_corr_data[i_Tu_array,i_pu_array,i_Qd_array,gauss]
        k_pd_Td_Qu = k_corr_data[i_Td_array,i_pd_array,i_Qu_array,gauss]
        k_pd_Tu_Qu = k_corr_data[i_Tu_array,i_pd_array,i_Qu_array,gauss]
        k_pu_Td_Qu = k_corr_data[i_Td_array,i_pu_array,i_Qu_array,gauss]
        k_pu_Tu_Qu = k_corr_data[i_Tu_array,i_pu_array,i_Qu_array,gauss]
    else :
        k_pd_Td_Qd = k_corr_data[i_pd_array,i_Td_array,i_Qd_array]
        k_pd_Tu_Qd = k_corr_data[i_pd_array,i_Tu_array,i_Qd_array]
        k_pu_Td_Qd = k_corr_data[i_pu_array,i_Td_array,i_Qd_array]
        k_pu_Tu_Qd = k_corr_data[i_pu_array,i_Tu_array,i_Qd_array]
        k_pd_Td_Qu = k_corr_data[i_pd_array,i_Td_array,i_Qu_array]
        k_pd_Tu_Qu = k_corr_data[i_pd_array,i_Tu_array,i_Qu_array]
        k_pu_Td_Qu = k_corr_data[i_pu_array,i_Td_array,i_Qu_array]
        k_pu_Tu_Qu = k_corr_data[i_pu_array,i_Tu_array,i_Qu_array]

    k_1 = k_pd_Td_Qd*coeff_6_array + k_pd_Td_Qu*coeff_5_array
    k_2 = k_pu_Td_Qd*coeff_6_array + k_pu_Td_Qu*coeff_5_array
    k_3 = k_pd_Tu_Qd*coeff_6_array + k_pd_Tu_Qu*coeff_5_array
    k_4 = k_pu_Tu_Qd*coeff_6_array + k_pu_Tu_Qu*coeff_5_array

    if Optimal == False :
        k_d = k_1*coeff_4_array + k_2*coeff_3_array
        k_u = k_3*coeff_4_array + k_4*coeff_3_array
        k_inter = k_d*coeff_2_array + k_u*coeff_1_array
    else :
        k_u = k_3*coeff_4_array + k_4*coeff_3_array
        b_m = coeff_1_array[0,zer_n]
        a_m = coeff_1_array[1,zer_n]
        T = coeff_1_array[2,zer_n]
        k_inter[zer_n] = k_u*a_m*np.exp(b_m/T)
        if zer_p.size != 0 :
            k_d = k_1*coeff_4_array + k_2*coeff_3_array
            k_inter[zer_p] = k_d*(1-coeff_1_array[1,zer_p]) + k_u*coeff_1_array[1,zer_p]

    return k_inter*0.0001


########################################################################################################################


def Rayleigh_scattering (P_array,T_array,bande_sample,x_mol_species,n_species,zero,rang,rang_ref,Kcorr=True,MarcIngo=False,oneD=False) :

    if Kcorr == True :
        dim_bande = bande_sample.size-1
    else :
        dim_bande = bande_sample.size

    k_sca_rmd = np.zeros((P_array.size,dim_bande))

    fact = 32*np.pi**3/(3*(101325/(R_gp*288)*N_A)**2)
    fact_2 = 24*np.pi**3/((101325/(R_gp*288)*N_A)**2)
    n_mol_tot = P_array/(R_gp*T_array)*N_A

    if zero.size != 0 :

        n_mol_tot[zero] = np.zeros((zero.size))

    if oneD == False :
        if rang == rang_ref :
            bar = ProgressBar(dim_bande,'Scattering computation progression')

    rank = 1

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            w_n = (bande_sample[i_bande] + bande_sample[i_bande + 1])/2.
        else :
            w_n = bande_sample[i_bande]

        wl = 1./(w_n*10**(2))

        for sp in range(n_species.size) :

            if str(n_species[sp]) == 'O2' :

                f_K = 1.096 + 1.385e-11*w_n**2 + 1.448e-20*w_n**4

                if w_n > 45250 :

                    index = (2.37967e-4 + 1.689884e+5/(4.09e+9-w_n**2))**2

                if w_n < 45250 and w_n > 34720 :

                    index = (2.21204e-4 + 2.03187e+5/(4.09e+9-w_n**2))**2

                if w_n < 34720 and w_n > 18315 :

                    index = (2.0564e-4 + 2.480899e+5/(4.09e+9-w_n**2))**2

                if w_n < 18315 :

                    index = (2.1351e-4 + 2.18567e+5/(4.09e+9-w_n**2))**2

                sig = fact/(wl**4)*index*f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'N2' :

                f_K = 1.034 + 3.17e-12*w_n**2

                if w_n < 21360 :

                    index = (6.498e-5 + (3.074e+6)/(14.4e+9-w_n**2))**2

                if w_n > 21360 and w_n < 39370 :

                    index = (6.677e-5 + (3.1882e+6)/(14.4e+9-w_n**2))**2

                if w_n > 39370 :

                    index = (6.9987e-5 + (3.23358e+6)/(14.4e+9-w_n**2))**2

                sig = fact/(wl**4)*index*f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'CO2' :

                f_K = 1.1364 + 2.53e-11*w_n**2
                index = 1 + 1.1427e-2*(5.79925e+3/(1.66175e+10-w_n**2)+1.2005e+2/(7.9608e+9-w_n**2)+5.3334/(5.6306e+9-w_n**2)+\
                                    4.3244/(4.619e+9-w_n**2)+0.12181e-4/(5.8474e+6-w_n**2))
                index = (index**2-1)/(index**2+2)
                sig = fact_2/(wl**4)*index**f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'CO' :

                f_K = 1.016
                index = (2.2851e-4 + (4.56e+3)/(71427**2-w_n**2))**2
                sig = fact/(wl**4)*index*f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'CH4' :

                f_K = 1.
                index = (4.6662e-4 + (4.02e-14)*w_n**2)**2
                sig = fact/(wl**4)*index*f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'Ar' :

                f_K = 1.
                index = (6.432135e-5 + (2.8606e+4)/(14.4e+9 - w_n**2))**2
                sig = fact/(wl**4)*index*f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'H2' :

                f_K = 1.

                if w_n > 33333 :

                    sig = f_K*8.45e-57/(wl**4)

                else :

                    sig = f_K*(8.14e-57/(wl**4) + 1.28e-70/(wl**6) + 1.61e-84/(wl**8))

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'He' :

                f_K = 1.

                if MarcIngo == True :

                    sig = f_K*(5.484e-58/(wl**4) + 1.33e-72/(wl**6))

                else :

                    index = (2.283e-5 + 1.8102e+5/(1.532e+10-w_n**2))**2
                    sig = fact/(wl**4)*index*f_K

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]


            if n_species[sp] == 'H2O' :

                f_K = (6+3*0.17)/(6-7*0.17)

                if w_n < 43480 :

                    index = 1 + 0.85*(5.7918e+10/(2.380185e+14 - 1/wl**2) + 1.67909e+9/(5.7362e+13 - 1/wl**2))

                else :

                    index = 1 + 0.85*(8.06051 + 2.48099e+10/(132.274e+14 - 1/wl**2) + 1.74557e+8/(3.932957e+13 - 1/wl**2))
                index = (index**2-1)/(index**2+2)
                sig = f_K/(wl**4)*index*fact_2

                k_sca_rmd[:,i_bande] += sig*n_mol_tot*x_mol_species[sp,:]

        if oneD == False :
            if rang == rang_ref :
                bar.animate(i_bande + 1)
                rank += 1


    return k_sca_rmd


########################################################################################################################


def cloud_scattering(Qext,bande_cloud,P,T,bande_sample,M,rho_p,gen,r_eff,r_cloud,zero,rang,rang_ref,Kcorr) :

    wh, = np.where(r_cloud == r_eff)

    if wh.size == 0 :

        whu, = np.where(r_cloud > r_eff)

        if whu.size != 0 :

            if whu[0] != 0 :

                i_r_u = whu[0]
                r_u = r_cloud[whu[0]]
                i_r_d = i_r_u - 1
                r_d = r_cloud[whu[0]-1]

                coeff1 = (r_eff - r_d)/(r_u - r_d)
                coeff2 = 1 - coeff1

            else :

                i_r_u,i_r_d = 0,0
                r_u,r_d = r_cloud[0], r_cloud[0]

                coeff1 = 1
                coeff2 = 0
        else :

            i_r_u,i_r_d = r_cloud.size - 1, r_cloud.size - 1
            r_u,r_d = r_cloud[r_cloud.size - 1],r_cloud[r_cloud.size - 1]

            coeff1 = 1
            coeff2 = 0

    else :

        i_r_u,i_r_d = wh[0], wh[0]
        r_u,r_d = r_cloud[wh[0]],r_cloud[wh[0]]

        coeff1 = 1
        coeff2 = 0

    k_cloud_rmd = np.zeros((P.size,bande_sample.size))

    Q_int = Qext[i_r_u,:]*coeff1 + Qext[i_r_d,:]*coeff2

    rank = 1

    if rang == rang_ref : 
        bar = ProgressBar(bande_sample.size,'Clouds scattering computation progression')

    for i_bande in range(bande_sample.size) :

        wh, = np.where(bande_cloud == bande_sample[i_bande])

        if wh.size == 0 :

            whu, = np.where(bande_cloud > bande_sample[i_bande])

            if whu.size != 0 :

                if whu[0] != 0 :

                    i_b_u = whu[0]
                    b_u = bande_cloud[whu[0]]
                    i_b_d = i_b_u - 1
                    b_d = bande_cloud[whu[0]-1]

                    coeff3 = (bande_sample[i_bande] - b_d)/(b_u - b_d)
                    coeff4 = 1 - coeff3

                else :

                    i_b_u,i_b_d = 0,0
                    b_u,b_d = bande_cloud[0], bande_cloud[0]

                    coeff3 = 1
                    coeff4 = 0
            else :

                i_b_u,i_b_d = bande_cloud.size - 1, bande_cloud.size - 1
                b_u,b_d = bande_cloud[bande_cloud.size - 1],bande_cloud[bande_cloud.size - 1]

                coeff3 = 1
                coeff4 = 0

        else :

            i_b_u,i_b_d = wh[0], wh[0]
            b_u,b_d = bande_cloud[wh[0]],bande_cloud[wh[0]]

            coeff3 = 1
            coeff4 = 0

        #print(i_bande,i_b_u,i_b_d,coeff1,coeff2,coeff3,coeff4)

        Q_fin = Q_int[i_b_u]*coeff3 + Q_int[i_b_d]*coeff4

        k_cloud_rmd[:,i_bande] = 3/4.*(Q_fin*gen*P*M/(rho_p*r_eff*R_gp*T))

        if rang == rang_ref :
            bar.animate(i_bande + 1)
            rank += 1

    if zero.size != 0 :

        k_cloud_rmd[zero,:] = np.zeros((zero.size,bande_sample.size))

    return k_cloud_rmd


########################################################################################################################


def refractive_index (P_array,T_array,bande_sample,x_mol_species,n_species,Kcorr=True,oneD=False) :

    if Kcorr == True :
        dim_bande = bande_sample.size-1
    else :
        dim_bande = bande_sample.size

    n_rmd = np.zeros((P_array.size,dim_bande))

    fact = 32*np.pi**3/(3*(101325/(R_gp*288)*N_A)**2)

    if oneD == False :
        bar = ProgressBar(dim_bande,'Refractive index computation progression')

    for i_bande in range(dim_bande) :

        if Kcorr == True :
            w_n = (bande_sample[i_bande] + bande_sample[i_bande + 1])/2.
        else :
            w_n = bande_sample

        wl = 1./(float(10**2*w_n))

        for sp in range(n_species.size) :

            if str(n_species[sp]) == 'O2' :

                if w_n > 45250 :

                    n_rmd[:,i_bande] += (1 + 2.37967e-4 + 1.689884e+5/(4.09e+9-w_n**2))*x_mol_species[sp,:]

                if w_n < 45250 and w_n > 34720 :

                    n_rmd[:,i_bande] += (1 + 2.21204e-4 + 2.03187e+5/(4.09e+9-w_n**2))*x_mol_species[sp,:]

                if w_n < 34720 and w_n > 18315 :

                    n_rmd[:,i_bande] += (1 + 2.0564e-4 + 2.480899e+5/(4.09e+9-w_n**2))*x_mol_species[sp,:]

                if w_n < 18315 :

                    n_rmd[:,i_bande] += (1 + 2.1351e-4 + 2.18567e+5/(4.09e+9-w_n**2))*x_mol_species[sp,:]

            if n_species[sp] == 'N2' :

                if w_n < 21360 :

                    n_rmd[:,i_bande] += (1+ 6.498e-5 + (3.074e+6)/(14.4e+9-w_n**2))*x_mol_species[sp,:]

                if w_n > 21360 and w_n < 39370 :

                    n_rmd[:,i_bande] += (1 + 6.677e-5 + (3.1882e+6)/(14.4e+9-w_n**2))*x_mol_species[sp,:]

                if w_n > 39370 :

                    n_rmd[:,i_bande] += (1 + 6.9987e-5 + (3.23358e+6)/(14.4e+9-w_n**2))*x_mol_species[sp,:]

            if n_species[sp] == 'CO2' :

                n_rmd[:,i_bande] += (1 + 1.1427e+6*(5.79925e+4/(5.0821e+13-w_n**2)+1.2005e+2/(7.9608e+9-w_n**2)+5.3334/(5.6306e+9-w_n**2)+\
                                    4.3244/(4.619e+9-w_n**2)+0.12181e-4/(5.8474e+6-w_n**2)))*x_mol_species[sp,:]

            if n_species[sp] == 'CO' :

                n_rmd[:,i_bande] += (1 + 2.2851e-4 + (4.56e+3)/(71427**2-w_n**2))*x_mol_species[sp,:]

            if n_species[sp] == 'CH4' :

                n_rmd[:,i_bande] += (1 + 4.6662e-4 + (4.02e-14)*w_n**2)*x_mol_species[sp,:]

            if n_species[sp] == 'Ar' :

                n_rmd[:,i_bande] += (1 + 6.432135e-5 + (2.8606e+4)/(14.4e+9 - w_n**2))*x_mol_species[sp,:]

            if n_species[sp] == 'H2' :

                if w_n > 33333 :

                    n_rmd[:,i_bande] += (1 + np.sqrt(8.45e-57/fact))*x_mol_species[sp,:]

                else :

                    n_rmd[:,i_bande] += (1 + np.sqrt(8.14e-57/fact + 1.28e-70/(fact*wl**2) + 1.61e-84/(fact*wl**4)))*x_mol_species[sp,:]

            if n_species[sp] == 'He' :

                n_rmd[:,i_bande] += (1 + 2.283e-5 + 1.8102e+5/(1.532e+10-w_n**2))*x_mol_species[sp,:]

            if n_species[sp] == 'H2O' :

                if w_n < 43480 :

                    n_rmd[:,i_bande] += (1 + 0.85*(5.7918e+10/(2.380185e+14 - 1/wl**2) + 1.67909e+9/(5.7362e+13 - 1/wl**2)))*x_mol_species[sp,:]

                else :

                    n_rmd[:,i_bande] += (1 + 0.85*(8.06051 + 2.48099e+10/(132.274e+14 - 1/wl**2) + 1.74557e+8/(3.932957e+13 - 1/wl**2)))*x_mol_species[sp,:]

        if oneD == False :
            bar.animate(i_bande + 1)

    return n_rmd


########################################################################################################################




########################################################################################################################


def k_cont_interp_h2h2_integration(K_cont_h2h2,wavelength_cont_h2h2,T_array,bande_array,T_cont_h2h2,rank,rank_ref,Kcorr=True) :


    losch = 2.6867774e19
    size = T_array.size
    if Kcorr == True :
        dim_bande = bande_array.size - 1
    else :
        dim_bande = bande_array.size

    k_interp_h2h2 = np.zeros((dim_bande,size))

    T_min = T_cont_h2h2[0]
    T_max = T_cont_h2h2[T_cont_h2h2.size-1]
    end = T_cont_h2h2.size - 1
    stop = 0

    if rank == rank_ref :
        bar = ProgressBar(dim_bande,'Continuum H2/H2 computation progression')

    for i_bande in range(dim_bande) :

        if Kcorr == True :

            k_interp = np.zeros(size)

        else :

            k_interp = np.zeros((2,size))

        if i_bande == 0 :

            i_Tu_array = np.zeros(size, dtype='int')
            coeff_array = np.zeros(size)

            if Kcorr == True :

                wave_max = bande_array[1]
                wave_min = bande_array[0]

                zone_wave, = np.where((wavelength_cont_h2h2 >= wave_min)*(wavelength_cont_h2h2 <= wave_max))
                #print("For bande number %i, we took into account wavelenght (cm-1) :" %(i_bande))
                #print(wavelength_cont[zone_wave])
                fact = zone_wave.size

                for i_wave in zone_wave :

                    if i_wave == zone_wave[0] :

                        for i in range(size) :

                            T = T_array[i]

                            if T < T_min or T > T_max :

                                if T < T_min :

                                    if T != 0 :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = K_cont_h2h2[0,i_wave]

                                    else :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = 0.

                                if T > T_max :

                                    i_Tu, i_Td, coeff = end, end, 1

                                    k_interp[i] = K_cont_h2h2[T_cont_h2h2.size-1,i_wave]

                            else :

                                wh, = np.where(T_cont_h2h2 > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2h2[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2h2[i_Td]

                                k_h2h2_Tu = K_cont_h2h2[i_Tu,i_wave]
                                k_h2h2_Td = K_cont_h2h2[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[i] = k_h2h2_Tu*coeff +  k_h2h2_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff

                    else :

                        zer, = np.where(i_Tu_array == 0)

                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)

                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp += k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))

                        k_interp_h2h2[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2h2 >= bande_array[i_bande])

                wave_up = wavelength_cont_h2h2[zone_wave_up[0]]
                wave_down = wavelength_cont_h2h2[zone_wave_up[0]-1]

                coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :

                    if i_wave == zone_wave_up[0]-1 :
                        if i_wave == -1 :
                            i_wave = 0

                        for i in range(size) :

                            T = T_array[i]

                            if T < T_min or T > T_max :

                                if T < T_min :

                                    if T != 0 :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = K_cont_h2h2[0,i_wave]

                                    else :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = 0.

                                if T > T_max :

                                    i_Tu, i_Td, coeff = end, end, 1

                                    k_interp[0,i] = K_cont_h2h2[T_cont_h2h2.size-1,i_wave]

                            else :

                                wh, = np.where(T_cont_h2h2 > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2h2[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2h2[i_Td]

                                k_h2h2_Tu = K_cont_h2h2[i_Tu,i_wave]
                                k_h2h2_Td = K_cont_h2h2[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[0,i] = k_h2h2_Tu*coeff +  k_h2h2_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff

                    else :

                        zer, = np.where(i_Tu_array == 0)

                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)

                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp[1,:] = k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                        k_interp_h2h2[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        else :

            if Kcorr == True :

                wave_max = bande_array[i_bande + 1]
                wave_min = bande_array[i_bande]

                zone_wave, = np.where((wavelength_cont_h2h2 >= wave_min)*(wavelength_cont_h2h2 <= wave_max))
                #print("For bande number %i, we took into account wavelenght (cm-1) :" %(i_bande+1))
                #print(wavelength_cont[zone_wave])
                fact = zone_wave.size

                for i_wave in zone_wave :

                    k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                    k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                    k_interp += k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))

                        k_interp_h2h2[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2h2 >= bande_array[i_bande])

                if zone_wave_up.size == 0 :

                    if stop == 0 :

                        i_wave = wavelength_cont_h2h2.size-1

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp_h2h2[i_bande,:] = k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                        stop += 1

                    else :

                        k_interp_h2h2[i_bande,:] = k_interp_h2h2[i_bande-1,:]

                else :

                    wave_up = wavelength_cont_h2h2[zone_wave_up[0]]
                    wave_down = wavelength_cont_h2h2[zone_wave_up[0]]

                    coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                    i = 0

                    for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :

                        k_h2h2_Tu = K_cont_h2h2[i_Tu_array,i_wave]
                        k_h2h2_Td = K_cont_h2h2[i_Td_array,i_wave]

                        k_interp[i,:] = k_h2h2_Tu*coeff_array + k_h2h2_Td*(1 - coeff_array)

                        i += 1

                    k_interp_h2h2[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        if rank == rank_ref :
            bar.animate(i_bande+1)

    return k_interp_h2h2*100*losch**2


########################################################################################################################


def k_cont_interp_h2he_integration(K_cont_h2he,wavelength_cont_h2he,T_array,bande_array,T_cont_h2he,rank,rank_ref,Kcorr=True) :

    losch = 2.6867774e19
    size = T_array.size
    if Kcorr == True :
        dim_bande = bande_array.size - 1
    else :
        dim_bande = bande_array.size
    k_interp_h2he = np.zeros((dim_bande,size))

    T_min = T_cont_h2he[0]
    T_max = T_cont_h2he[T_cont_h2he.size-1]
    end = T_cont_h2he.size - 1
    stop = 0

    if rank == rank_ref :
        bar = ProgressBar(dim_bande, 'Continuum H2/He computation progression')

    for i_bande in range(dim_bande) :

        if Kcorr == True :

            k_interp = np.zeros(size)

        else :

            k_interp = np.zeros((2,size))

        if i_bande == 0 :

            i_Tu_array = np.zeros(size, dtype='int')
            coeff_array = np.zeros(size)

            if Kcorr == True :

                wave_max = bande_array[1]
                wave_min = bande_array[0]

                zone_wave, = np.where((wavelength_cont_h2he >= wave_min)*(wavelength_cont_h2he <= wave_max))
                #print("For bande number %i, we took into account wavelenght (cm-1) :" %(i_bande))
                #print(wavelength_cont[zone_wave])
                fact = zone_wave.size

                for i_wave in zone_wave :

                    if i_wave == zone_wave[0] :

                        for i in range(size) :

                            T = T_array[i]

                            if T < T_min or T > T_max :

                                if T < T_min :

                                    if T != 0 :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = K_cont_h2he[0,i_wave]

                                    else :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = 0.

                                if T > T_max :

                                    i_Tu, i_Td, coeff = end, end, 1

                                    k_interp[i] = K_cont_h2he[T_cont_h2he.size-1,i_wave]

                            else :

                                wh, = np.where(T_cont_h2he > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2he[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2he[i_Td]

                                k_h2he_Tu = K_cont_h2he[i_Tu,i_wave]
                                k_h2he_Td = K_cont_h2he[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[i] = k_h2he_Tu*coeff +  k_h2he_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff

                    else :

                        zer, = np.where(i_Tu_array == 0)

                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)

                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp += k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))

                        k_interp_h2he[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2he >= bande_array[i_bande])

                wave_up = wavelength_cont_h2he[zone_wave_up[0]]
                wave_down = wavelength_cont_h2he[zone_wave_up[0]-1]

                coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :

                    if i_wave == zone_wave_up[0]-1 :

                        if i_wave == -1 :
                            i_wave = 0

                        for i in range(size) :

                            T = T_array[i]

                            if T < T_min or T > T_max :

                                if T < T_min :

                                    if T != 0 :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = K_cont_h2he[0,i_wave]

                                    else :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = 0.

                                if T > T_max :

                                    i_Tu, i_Td, coeff = end, end, 1

                                    k_interp[0,i] = K_cont_h2he[T_cont_h2he.size-1,i_wave]

                            else :

                                wh, = np.where(T_cont_h2he > T)
                                i_Tu = wh[0]
                                Tu = T_cont_h2he[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_h2he[i_Td]

                                k_h2he_Tu = K_cont_h2he[i_Tu,i_wave]
                                k_h2he_Td = K_cont_h2he[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[0,i] = k_h2he_Tu*coeff +  k_h2he_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff

                    else :

                        zer, = np.where(i_Tu_array == 0)

                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)

                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp[1,:] = k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                        k_interp_h2he[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        else :

            if Kcorr == True :

                wave_max = bande_array[i_bande + 1]
                wave_min = bande_array[i_bande]

                zone_wave, = np.where((wavelength_cont_h2he >= wave_min)*(wavelength_cont_h2he <= wave_max))
                #print("For bande number %i, we took into account wavelenght (cm-1) :" %(i_bande+1))
                #print(wavelength_cont[zone_wave])
                fact = zone_wave.size

                for i_wave in zone_wave :

                    k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                    k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                    k_interp += k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))

                        k_interp_h2he[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_h2he >= bande_array[i_bande])

                if zone_wave_up.size == 0 :

                    if stop == 0 :

                        i_wave = wavelength_cont_h2he.size-1

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp_h2he[i_bande,:] = k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                        stop += 1

                    else :

                        k_interp_h2he[i_bande,:] = k_interp_h2he[i_bande-1,:]

                else :

                    wave_up = wavelength_cont_h2he[zone_wave_up[0]]
                    wave_down = wavelength_cont_h2he[zone_wave_up[0]]

                    coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                    i = 0

                    for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :

                        k_h2he_Tu = K_cont_h2he[i_Tu_array,i_wave]
                        k_h2he_Td = K_cont_h2he[i_Td_array,i_wave]

                        k_interp[i,:] = k_h2he_Tu*coeff_array + k_h2he_Td*(1 - coeff_array)

                        i += 1

                    k_interp_h2he[i_bande,:] = k_interp[1,:]*coef + k_interp[1,:]*(1 - coef)

        if rank == rank_ref :
            bar.animate(i_bande+1)

    return k_interp_h2he*100*losch**2


########################################################################################################################


def k_cont_interp_spespe_integration(K_cont_spespe,wavelength_cont_spespe,T_array,bande_array,T_cont_spespe,rank,rank_ref,species,Kcorr=True,H2O=False) :

    losch = 2.6867774e19
    size = T_array.size
    if Kcorr == True :
        dim_bande = bande_array.size - 1
    else :
        dim_bande = bande_array.size
    k_interp_spespe = np.zeros((dim_bande,size))

    T_min = T_cont_spespe[0]
    T_max = T_cont_spespe[T_cont_spespe.size-1]
    end = T_cont_spespe.size - 1
    stop = 0

    if rank == rank_ref :
        bar = ProgressBar(dim_bande,'Continuum %s computation progression'%(species))

    for i_bande in range(dim_bande) :

        if Kcorr == True :

            k_interp = np.zeros(size)

        else :

            k_interp = np.zeros((2,size))

        if i_bande == 0 :

            i_Tu_array = np.zeros(size, dtype='int')
            coeff_array = np.zeros(size)

            if Kcorr == True :

                wave_max = bande_array[1]
                wave_min = bande_array[0]

                zone_wave, = np.where((wavelength_cont_spespe >= wave_min)*(wavelength_cont_spespe <= wave_max))
                #print("For bande number %i, we took into account wavelenght (cm-1) :" %(i_bande))
                fact = zone_wave.size

                for i_wave in zone_wave :

                    if i_wave == zone_wave[0] :

                        for i in range(size) :

                            T = T_array[i]

                            if T < T_min or T > T_max :

                                if T < T_min :

                                    if T != 0 :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = K_cont_spespe[0,i_wave]

                                    else :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[i] = 0.

                                if T > T_max :

                                    i_Tu, i_Td, coeff = end, end, 1

                                    k_interp[i] = K_cont_spespe[T_cont_spespe.size-1,i_wave]

                            else :

                                wh, = np.where(T_cont_spespe > T)
                                i_Tu = wh[0]
                                Tu = T_cont_spespe[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_spespe[i_Td]

                                k_spespe_Tu = K_cont_spespe[i_Tu,i_wave]
                                k_spespe_Td = K_cont_spespe[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[i] = k_spespe_Tu*coeff +  k_spespe_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff

                    else :

                        zer, = np.where(i_Tu_array == 0)

                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')
                        print 'pouet'

                        ex, = np.where(i_Tu_array == end)

                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp += k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))

                        k_interp_spespe[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_spespe >= bande_array[i_bande])

                wave_up = wavelength_cont_spespe[zone_wave_up[0]]
                wave_down = wavelength_cont_spespe[zone_wave_up[0]-1]

                coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :

                    if i_wave == zone_wave_up[0]-1 :

                        if i_wave == -1 :
                            i_wave = 0

                        for i in range(size) :

                            T = T_array[i]

                            if T < T_min or T > T_max :

                                if T < T_min :

                                    if T != 0 :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = K_cont_spespe[0,i_wave]

                                    else :

                                        i_Tu, i_Td, coeff = 0, 0, 0
                                        k_interp[0,i] = 0.

                                if T > T_max :

                                    i_Tu, i_Td, coeff = end, end, 1

                                    k_interp[0,i] = K_cont_spespe[T_cont_spespe.size-1,i_wave]

                            else :

                                wh, = np.where(T_cont_spespe > T)
                                i_Tu = wh[0]
                                Tu = T_cont_spespe[i_Tu]
                                i_Td = i_Tu - 1
                                Td = T_cont_spespe[i_Td]

                                k_spespe_Tu = K_cont_spespe[i_Tu,i_wave]
                                k_spespe_Td = K_cont_spespe[i_Td,i_wave]

                                coeff = (T - Td)/(float(Tu + Td))

                                k_interp[0,i] = k_spespe_Tu*coeff +  k_spespe_Td*(1 - coeff)

                            i_Tu_array[i] = i_Tu
                            i_Td_array = i_Tu_array - 1
                            coeff_array[i] = coeff

                    else :

                        zer, = np.where(i_Tu_array == 0)

                        i_Td_array[zer] = np.zeros(zer.size, dtype='int')

                        ex, = np.where(i_Tu_array == end)

                        i_Td_array[ex] = np.ones(ex.size, dtype='int')*end

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp[1,:] = k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                        k_interp_spespe[i_bande,:] = k_interp[1,:]*coef + k_interp[0,:]*(1 - coef)

        else :

            if Kcorr == True :

                wave_max = bande_array[i_bande + 1]
                wave_min = bande_array[i_bande]

                zone_wave, = np.where((wavelength_cont_spespe >= wave_min)*(wavelength_cont_spespe <= wave_max))
                #print("For bande number %i, we took into account wavelenght (cm-1) :" %(i_bande+1))
                #print(wavelength_cont[zone_wave])
                fact = zone_wave.size

                for i_wave in zone_wave :

                    k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                    k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                    k_interp += k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                    if i_wave == zone_wave[fact-1] :

                        k_interp = k_interp/(float(fact))

                        k_interp_spespe[i_bande,:] = k_interp

            else :

                zone_wave_up, = np.where(wavelength_cont_spespe >= bande_array[i_bande])

                if zone_wave_up.size == 0 :

                    if stop == 0 :

                        i_wave = wavelength_cont_spespe.size-1

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp_spespe[i_bande,:] = k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                        stop += 1

                    else :

                        k_interp_spespe[i_bande,:] = k_interp_spespe[i_bande-1,:]

                else :

                    wave_up = wavelength_cont_spespe[zone_wave_up[0]]
                    wave_down = wavelength_cont_spespe[zone_wave_up[0]]

                    coef = (bande_array[i_bande] - wave_down)/(float(wave_up + wave_down))

                    i = 0

                    for i_wave in [zone_wave_up[0]-1,zone_wave_up[0]] :

                        k_spespe_Tu = K_cont_spespe[i_Tu_array,i_wave]
                        k_spespe_Td = K_cont_spespe[i_Td_array,i_wave]

                        k_interp[i,:] = k_spespe_Tu*coeff_array + k_spespe_Td*(1 - coeff_array)

                        i += 1

                    k_interp_spespe[i_bande,:] = k_interp[1,:]*coef + k_interp[1,:]*(1 - coef)

        if rank == rank_ref :
            bar.animate(i_bande+1)

    if H2O == True :
        return k_interp_spespe*0.0001
    else :
        return k_interp_spespe*100*losch**2
