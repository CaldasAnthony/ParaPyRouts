from GJ1214b_script_para import *
from mpi4py import MPI

########################################################################################################################
########################################################################################################################

# Initialisation de la parallelisation

comm = MPI.COMM_WORLD
rank = comm.rank
number_rank = comm.size

########################################################################################################################

reso_alt = int(h/1000)
reso_long = int(reso_long)
reso_lat = int(reso_lat)

if rank == 0 :
    
    message_clouds = ''
    if Cloudy == True :
        for i in range(c_species.size) :
            message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff*10**6,rho_p[i]/1000.)
        print 'Clouds in the atmosphere (grain radius/density) : %s'%(message_clouds)
    else :
        print 'There is no clouds'
    print 'Width of layers : %i m'%(delta_z)
    print 'Top of the atmosphere : %i m'%(h)
    print 'Mean radius of the exoplanet : %i m'%(Rp)
    print 'Mean surface gravity : %.2f m/s^2'%(g0)
    print 'Mean molar mass : %.5f kg/mol'%(M)
    print 'Extrapolation type for the upper atmosphere : %s'%(Upper)
    number = 2 + m_species.size + c_species.size + n_species.size + 1
    print 'Resolution of the GCM simulation (latitude/longitude) : %i/%i'%(reso_lat,reso_long)

########################################################################################################################





########################################################################################################################
########################################################################################################################
###########################################      PARAMETERS      #######################################################
########################################################################################################################
########################################################################################################################

# Telechargement des sections efficaces ou des k_correles

########################################################################################################################

if Profil == True :
    
    T_comp = np.load("%s%s/T_comp.npy"%(path,name_source))
    P_comp = np.load("%s%s/P_comp.npy"%(path,name_source))
    if Tracer == True :
        Q_comp = np.load("%s%s/Q_comp.npy"%(path,name_source))
    else :
        Q_comp = np.array([])
    X_species = np.load("%s%s/x_species_comp.npy"%(path,name_source))

                                    ###### Parallele encoding init ######

    sha = np.zeros(4,dtype=np.int)
    if rank == 0 :
        print 'Record of GCM data start'
        P, T, Q, gen = Boxes_spheric_data(data_base,t_selec,c_species,Surf,Tracer,Cloudy,TimeSelec)
        print 'Record of GCM finished with success'
        sha = np.shape(P)
        sha = np.array(sha,dtype=np.int)
        P = np.array(P,dtype=np.float64)
    comm.Bcast([sha,MPI.INT],root=0)
    if rank != 0 :
        P = np.zeros(sha,dtype=np.float64)
    comm.Bcast([P,MPI.DOUBLE],root=0)
    if rank != 0 :
        T = np.zeros(sha,dtype=np.float64)
    comm.Bcast([T,MPI.DOUBLE],root=0)
    if Tracer == True :
        if rank != 0 :
            Q = np.zeros(sha, dtype=np.float64)
        comm.Bcast([Q,MPI.DOUBLE],root=0)
    else :
        Q = np.array([])
    if Cloudy == True :
        if rank != 0 :
            gen = np.zeros((c_species.size,sha[0],sha[1],sha[2],sha[3]),dtype=np.float64)
        comm.Bcast([gen,MPI.DOUBLE],root=0)
    else :
        gen = np.array([])

    tss,pss,lass,loss = np.shape(P)
    dom_rank = repartition(lass,number_rank,rank,True)

    comm.Barrier()
    
    P_n, T_n = P[:,:,dom_rank,:], T[:,:,dom_rank,:]
    if Tracer == True :
        Q_n = Q[:,:,dom_rank,:]
    else : 
        Q_n = np.array([])
    if rank == 0 :
        print 'Interpolation for the composition start on %i processus'%(number_rank)

                                    ###### Parallele encoding end ######

    compo_i, M_i, z_i = Boxes_interlopation(P_n,T_n,Q_n,Rp,g0,M_species,number,P_comp,T_comp,Q_comp,n_species,X_species,M_species,\
            c_species,ratio_HeH2,Tracer,Cloudy,Composition,LogInterp)

                                    ###### Parallele encoding init ######

    comm.Barrier()
        
    if rank == 0 :
        print 'Interpolation for the composition finished with success'

    for r_n in range(number_rank) :
        if r_n != 0  and r_n == rank :
            comm.Send([compo_i,MPI.DOUBLE],dest=0,tag=0)
            comm.Send([M_i,MPI.DOUBLE],dest=0,tag=1)
            comm.Send([z_i,MPI.DOUBLE],dest=0,tag=2)
        elif r_n == 0 and rank == 0 :
            composition = np.zeros((n_species.size,tss,pss,lass,loss))
            M_molar = np.zeros((tss,pss,lass,loss),dtype=np.float64)
            z_sphe = np.zeros((tss,pss,lass,loss),dtype=np.float64)
            composition[:,:,:,dom_rank,:] = compo_i
            M_molar[:,:,dom_rank,:] = M_i
            z_sphe[:,:,dom_rank,:] = z_i
        elif r_n != 0 and rank == 0 :
            new_dom_rank = repartition(lass,number_rank,r_n,True)
            compo_n = np.zeros((n_species.size,tss,pss,new_dom_rank.size,loss),dtype=np.float64)
            M_n = np.zeros((tss,pss,new_dom_rank.size,loss),dtype=np.float64)
            z_n = np.zeros((tss,pss,new_dom_rank.size,loss),dtype=np.float64)
            comm.Recv([compo_n,MPI.DOUBLE],source=r_n,tag=0)
            comm.Recv([M_n,MPI.DOUBLE],source=r_n,tag=1)
            comm.Recv([z_n,MPI.DOUBLE],source=r_n,tag=2)
            composition[:,:,:,new_dom_rank,:] = compo_n
            M_molar[:,:,new_dom_rank,:] = M_n
            z_sphe[:,:,new_dom_rank,:] = z_n
    comm.Barrier()

    del compo_i, M_i, z_i
    info = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)

    if rank == 0 :

        np.save('/home/caldas/z',z_sphe)
        if h < np.amax(z_sphe) :
            h = np.amax(z_sphe)
            hmax = h
        else :
            hmax = np.amax(z_sphe)
        dim = int(h/delta_z)+1
        M_mean = np.nansum(M_molar[:,pss-1,:,:])/(tss*loss*lass)
        T_mean = np.nansum(T[:,pss-1,:,:])/(tss*loss*lass)
        T_min, T_max = np.amin(T[:,pss-1,:,:]), np.amax(T[:,pss-1,:,:])
        P_mean = np.nansum(P[:,pss-1,:,:])/(tss*loss*lass)
        g_roof = g0*1/(1+hmax/Rp)**2
        H_mean = R_gp*T_mean/(M_mean*g_roof)

        print "The thickness of the simulation is %i m"%(hmax)
        print "The thickness of the atmosphere is %i m"%(h)
        print "The scale height at the roof is %f m"%(H_mean)

        alp_h = H_mean*np.log(P_mean/P_h)
        z_h = np.amax(z_sphe) + alp_h/(1.+alp_h/(Rp+np.amax(z_sphe)))
        if TopPressure == True :
            dim = int(z_h/delta_z)+2
            z_h = (dim-1)*delta_z
            h = z_h
        if N_fixe == True :
            delta_z = h/np.float(n_layers)
            r_step, x_step = delta_z, delta_z
            dim = n_layers + 1
            print 'Number of layers imposed : %i'%(n_layers)
        else :
            n_layers = dim - 1
            print 'Number of layers : %i'%(n_layers)

        print "The final thickness of the atmosphere is %i m"%((dim-1)*delta_z)
        print "The final thickness of a layer is %i m"%(delta_z)
        print 'Conversion of the dataset will start soon'

        info = np.array([h,hmax,dim,delta_z,r_step,x_step,n_layers,T_mean,T_max,T_min], dtype=np.float64)

    comm.Bcast([info,MPI.DOUBLE],root=0)
    h, hmax, dim, delta_z, r_step, x_step, n_layers, T_mean, T_max, T_min = \
        info[0], info[1], np.int(info[2]), info[3], info[4], info[5], np.int(info[6]), info[7], info[8], info[9]
    reso_alt = int(h/delta_z)

    comm.Barrier()

    if Box == True : 
    
        if rank < lass/2 : 

            if number_rank >= lass/2 :
                number_rank = lass/2

            dom_rank = repartition(lass,number_rank,rank,True)

        if rank != 0 :
            z_sphe = np.zeros((tss,pss,lass,loss),dtype=np.float64)
            composition = np.zeros((n_species.size,tss,pss,lass,loss),dtype=np.float64)
            M_molar = np.zeros((tss,pss,lass,loss),dtype=np.float64)
        comm.Bcast([z_sphe,MPI.DOUBLE],root=0)
        comm.Bcast([composition,MPI.DOUBLE],root=0)
        comm.Bcast([M_molar,MPI.DOUBLE],root=0)

        if rank < lass/2 : 

            z_sphe = z_sphe[:,:,dom_rank,:]
            composition = composition [:,:,:,dom_rank,:]
            M_molar = M_molar[:,:,dom_rank,:]
            P = P[:,:,dom_rank,:]
            T = T[:,:,dom_rank,:]
            if Tracer == True :
                Q = Q[:,:,dom_rank,:]
            if Cloudy == True :
                gen = gen[:,:,:,dom_rank,:]

                                    ###### Parallele encoding end ######

            data_convert_part = Boxes_conversion(P,T,Q,gen,z_sphe,composition,delta_z,Rp,h,hmax,dim,g0,M_molar,number,T_comp,P_comp,\
                 Q_comp,X_species,M_species,ratio_HeH2,rank,Upper,Tracer,Cloudy,Composition,Middle,LogInterp)

                                    ###### Parallele encoding init ######

            for r_n in range(number_rank) :
                if r_n != 0  and r_n == rank :
                    comm.Send([data_convert_part,MPI.DOUBLE],dest=0,tag=r_n)
                elif r_n == 0 and r_n == rank :
                    data_convert = np.zeros((number,tss,dim,lass,loss))
                    data_convert[:,:,:,dom_rank,:] = data_convert_part
                elif r_n != 0 and rank == 0 :
                    new_dom_rank = repartition(lass,number_rank,r_n,True)
                    data_convert_part_n = np.zeros((number,tss,dim,new_dom_rank.size,loss), dtype=np.float64)
		    comm.Recv([data_convert_part_n,MPI.DOUBLE],source=r_n,tag=r_n)
		    data_convert[:,:,:,new_dom_rank,:] = data_convert_part_n

            del data_convert_part
	    if Tracer == True :
	        del Q
	    if Cloudy == True : 
	        del gen
	
        number_rank = comm.size
        comm.Barrier()
		
        del z_sphe, M_molar, P, T
	
        if rank == 0 : 
            del data_convert_part_n

        if rank == 0 :
            print 'Conversion of the dataset finished with success'
            np.save("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat),\
                    data_convert)

########################################################################################################################

if Parameters == True : 

    if Cylindre == True :
	
	z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)

        p_grid_n,q_grid_n,z_grid_n,n_level_rank = cylindric_assymatrix_parameter(Rp,h,alpha_step,delta_step,r_step,theta_step,theta_number,\
                                x_step,z_array,phi_rot,phi_obli,reso_long,reso_lat,rank,number_rank,Obliquity,Middle)
	
	if rank == 0 :  
	    sh_grid = np.shape(p_grid_n)
	    p_grid = np.ones((dim+5,sh_grid[1],sh_grid[2]),dtype=np.int)*(-1)
	    p_grid[n_level_rank,:,:] = p_grid_n
	    
	comm.Barrier()
	
	for r_n in range(number_rank) :
	    if rank != 0 and r_n == rank : 
	        sh_grid = np.array(np.shape(p_grid_n),dtype=np.int)
		comm.Send([sh_grid,MPI.INT],dest=0,tag=3)
		comm.Send([n_level_rank,MPI.INT],dest=0,tag=4)
	        comm.Send([p_grid_n,MPI.INT],dest=0,tag=5)
	    elif rank == 0 and r_n != 0 : 
		sh_grid_ne = np.zeros(3,dtype=np.int)
	        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=3)
		n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
	        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=4)
		p_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
		comm.Recv([p_grid_ne,MPI.INT],source=r_n,tag=5)
		p_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = p_grid_ne
		    
	if rank == 0 : 
            np.save("%s%s/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),p_grid)
	    del p_grid, p_grid_ne
	del p_grid_n
	
	comm.Barrier()
	
	for r_n in range(number_rank) :
	    if rank != 0 and r_n == 0 : 
	        sh_grid = np.array(np.shape(q_grid_n),dtype=np.int)
		comm.Send([sh_grid,MPI.INT],dest=0,tag=20)
		comm.Send([n_level_rank,MPI.INT],dest=0,tag=21)
		comm.Send([q_grid_n,MPI.INT],dest=0,tag=22)
	    elif rank == 0 and r_n == 0 : 
		sh_grid = np.shape(q_grid_n)
		q_grid = np.ones((dim+5,sh_grid[1],sh_grid[2]),dtype=np.int)*(-1)
	        q_grid[n_level_rank,:,:] = q_grid_n
	    elif rank == 0 and r_n !=0 : 
		sh_grid_ne = np.zeros(3,dtype=np.int)
	        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=20)
		n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
	        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=21)
		q_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
		comm.Recv([q_grid_ne,MPI.INT],source=r_n,tag=22)
		q_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = q_grid_ne
	
	if rank == 0 : 
            np.save("%s%s/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),q_grid)
	    del q_grid, q_grid_ne
	del q_grid_n
	
	comm.Barrier()
	
	for r_n in range(number_rank) :
	    if rank != 0 and r_n == 0 : 
	        sh_grid = np.array(np.shape(z_grid_n),dtype=np.int)
		comm.Send([sh_grid,MPI.INT],dest=0,tag=10)
		comm.Send([n_level_rank,MPI.INT],dest=0,tag=11)
	        comm.Send([z_grid_n,MPI.INT],dest=0,tag=12)
	    elif rank == 0 and r_n == 0 : 
		sh_grid = np.shape(z_grid_n)
		z_grid = np.ones((dim+5,sh_grid[1],sh_grid[2]),dtype=np.int)*(-1)
	        z_grid[n_level_rank,:,:] = z_grid_n
	    elif r_n != 0 and rank == 0 : 
		sh_grid_ne = np.zeros(3,dtype=np.int)
	        comm.Recv([sh_grid_ne,MPI.INT],source=r_n,tag=10)
		n_level_rank_ne = np.zeros(sh_grid_ne[0],dtype=np.int)
	        comm.Recv([n_level_rank_ne,MPI.INT],source=r_n,tag=11)
		z_grid_ne = np.zeros((sh_grid_ne),dtype=np.int)
		comm.Recv([z_grid_ne,MPI.INT],source=r_n,tag=12)
		z_grid[n_level_rank_ne,:,:sh_grid_ne[2]] = z_grid_ne
	
	if rank == 0 : 
            np.save("%s%s/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),z_grid)
	    del z_grid, z_grid_ne
	del z_grid_n
	
	if rank == 0 :
            print 'Computation of the cylindrical stictch finished with success'
	

########################################################################################################################

    if Corr == True :

        extra = 5
	n_lay_rank = repartition(n_layers,number_rank,rank,False)
	if n_layers%number_rank != 0 : 
	    n_extra = n_layers%number_rank-1
	else : 
	    n_extra = number_rank-1
	if rank == n_extra : 
	    for i_extra in range(extra) : 
	        n_lay_rank = np.append(n_lay_rank,n_layers+i_extra)
		    
	p_grid = np.load("%s%s/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
	p_grid = p_grid[n_lay_rank,:,:]
        q_grid = np.load("%s%s/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
	q_grid = q_grid[n_lay_rank,:,:]
        z_grid = np.load("%s%s/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
	z_grid = z_grid[n_lay_rank,:,:]

        data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,\
                reso_long,reso_lat))
	
	dx_grid_n,dx_grid_opt_n,order_grid_n,pdx_grid_n = dx_correspondance(p_grid,q_grid,z_grid,data_convert,x_step,r_step,\
                    theta_step,Rp,g0,h,t,reso_long,reso_lat,n_lay_rank,Middle,Integral,Discret,Gravity,Ord)

	comm.Barrier()
	
        for r_n in range(number_rank) :
	    if r_n == 0 and rank == 0 : 
	        length = np.zeros(number_rank,dtype=np.int)
		length[0] = np.shape(dx_grid_n)[2]
	    elif r_n == 0 and rank != 0 :
	        sh_dx_n = np.array(np.shape(dx_grid_n),dtype=np.int)
	        comm.Send([sh_dx_n,MPI.INT],dest=0,tag=0)
	    elif r_n != 0 and rank == 0 : 
	        sh_dx = np.zeros(3,dtype=np.int)
		comm.Recv([sh_dx,MPI.INT],source=r_n,tag=0)
		length[r_n] = sh_dx[2]
	
	comm.Barrier()
	
	if rank == 0 : 
	    x_size = np.amax(length)
	    dx_grid = np.ones((n_layers+extra,theta_number,x_size),dtype=np.int)*(-1)
	    order_grid = np.ones((6,n_layers+extra,theta_number,x_size),dtype=np.int)*(-1)
	    dx_grid[n_lay_rank,:,:length[0]] = dx_grid_n
	    order_grid[:,n_lay_rank,:,:length[0]] = order_grid_n
	    rank_size = np.array([], dtype=np.int)
	    rank_size = np.append(rank_size,n_lay_rank.size)
	
	for r_n in range(number_rank) : 
	    if r_n == rank and rank != 0 :
	        order_grid_n = np.array(order_grid_n,dtype=np.int)
	        comm.Send([dx_grid_n,MPI.INT],dest=0,tag=rank+1)
		comm.Send([order_grid_n,MPI.INT],dest=0,tag=rank+2)
	    elif r_n != 0 and rank == 0 :
		n_lay_rank_ne = repartition(n_layers,number_rank,r_n,False)
		if r_n == n_extra : 
		    for i_extra in range(extra) : 
	                n_lay_rank_ne = np.append(n_lay_rank_ne,n_layers+i_extra)
	        dx_grid_ne = np.zeros((n_lay_rank_ne.size,theta_number,length[r_n]),dtype=np.int)
		comm.Recv([dx_grid_ne,MPI.INT],source=r_n,tag=r_n+1)
		order_grid_ne = np.zeros((6,n_lay_rank_ne.size,theta_number,length[r_n]),dtype=np.int)
		comm.Recv([order_grid_ne,MPI.INT],source=r_n,tag=r_n+2)
		dx_grid[n_lay_rank_ne,:,:length[r_n]] = dx_grid_ne
		order_grid[:,n_lay_rank_ne,:,:length[r_n]] = order_grid_ne
		rank_size = np.append(rank_size,n_lay_rank_ne.size)
	
	if rank == 0 : 
	    np.save("%s%s/%s/dx_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli),dx_grid)
            np.save("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli),order_grid)
	    del dx_grid, dx_grid_ne
	    del order_grid, order_grid_ne
	del dx_grid_n
	del order_grid_n
	
	comm.Barrier()

        if Discret == True :
	    if rank == 0 : 
	        dx_grid_opt = np.ones((n_layers+extra,theta_number,x_size),dtype=np.float64)*(-1)
	        dx_grid_opt[n_lay_rank,:,:length[0]] = dx_grid_opt_n
		
	    for r_n in range(number_rank) : 
	        if r_n == rank and rank != 0 :
		    dx_grid_opt_n = np.array(dx_grid_opt_n, dtype=np.float64)
	            comm.Send([dx_grid_opt_n,MPI.DOUBLE],dest=0,tag=rank+11)
	        elif r_n != 0 and rank == 0 : 
	            dx_grid_opt_ne = np.zeros((rank_size[r_n],theta_number,length[r_n]),dtype=np.float64)
		    comm.Recv([dx_grid_opt_ne,MPI.DOUBLE],source=r_n,tag=r_n+11)
                    n_lay_rank_ne = repartition(n_layers,number_rank,r_n,False)
		    if r_n == n_extra : 
		        for i_extra in range(extra) : 
	                    n_lay_rank_ne = np.append(n_lay_rank_ne,n_layers+i_extra)
		    dx_grid_opt[n_lay_rank_ne,:,:length[r_n]] = dx_grid_opt_ne
             
	    if rank == 0 : 
	        np.save("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),dx_grid_opt)
		del dx_grid_opt, dx_grid_opt_ne
	    del dx_grid_opt_n
	    
	    comm.Barrier()
	    
        if Integral == True :
	    if rank == 0 : 
	        pdx_grid = np.ones((n_layers+extra,theta_number,x_size),dtype=np.float64)*(-delta_z)
	        pdx_grid[n_lay_rank,:,:length[0]] = pdx_grid_n
		
	    for r_n in range(number_rank) : 
	        if r_n == rank and rank != 0 :
		    pdx_grid_n = np.array(pdx_grid_n, dtype=np.float64)
	            comm.Send([pdx_grid_n,MPI.DOUBLE],dest=0,tag=rank+21)
	        elif r_n != 0 and rank == 0 : 
	            pdx_grid_ne = np.zeros((rank_size[r_n],theta_number,length[r_n]),dtype=np.float64)
		    comm.Recv([pdx_grid_ne,MPI.DOUBLE],source=r_n,tag=r_n+21)
		    n_lay_rank_ne = repartition(n_layers,number_rank,r_n,False)
		    if r_n == n_extra : 
		        for i_extra in range(extra) : 
	                    n_lay_rank_ne = np.append(n_lay_rank_ne,n_layers+i_extra)
		    pdx_grid[n_lay_rank_ne,:,:length[r_n]] = pdx_grid_ne
            
	    if rank == 0 : 
	        np.save("%s%s/%s/pdx_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),pdx_grid)
		del pdx_grid, pdx_grid_ne
	    del pdx_grid_n
	
	if rank == 0 : 
	    print 'Computation of optical pathes finished with success'

########################################################################################################################

    if Matrix == True :

        extra = 5
	n_lay_rank = repartition(n_layers,number_rank,rank,False)
	if n_layers%number_rank != 0 : 
	    n_extra = n_layers%number_rank-1
	else : 
	    n_extra = number_rank-1
	if rank == n_extra : 
	    for i_extra in range(extra) : 
	        n_lay_rank = np.append(n_lay_rank,n_layers+i_extra)
	
	data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                reso_lat))

        order_grid = np.load("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,\
                reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
	
	order_grid = order_grid[:,n_lay_rank,:,:]

        result_n = atmospheric_matrix_3D(order_grid,data_convert,t,Rp,c_species,rank,Tracer,Cloudy,Composition)

        if Tracer == True : 
	    m_m = 1
	else : 
	    m_m = 0
	if Cloudy == True : 
	    c_c = 1
	else : 
	    c_c = 0
	
        if rank == 0 : 
	    sh_res = np.shape(result_n)
	    result_P = np.zeros((n_layers + 5,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
	    result_T = np.zeros((n_layers + 5,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
	    result_Cn = np.zeros((n_layers + 5,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
	    result_P[n_lay_rank,:,:] = result_n[0]
	    result_T[n_lay_rank,:,:] = result_n[1]
	    result_Cn[n_lay_rank,:,:] = result_n[2]
	    if Tracer == True :
	        result_Q = np.zeros((n_layers + 5,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
		result_Q[n_lay_rank,:,:] = result_n[3]
	    if Cloudy == True : 
	        result_gen = np.zeros((c_species.size,n_layers + 5,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
		result_gen[:,n_lay_rank,:,:] = result_n[3+m_m]
	    if Composition == True : 
	        result_comp = np.zeros((n_species.size + 1, n_layers + 5,theta_number, np.shape(result_n[0])[2]), dtype=np.float64)
		result_comp[:,n_lay_rank,:,:] = result_n[3+m_m+c_c]
		
	length = np.shape(order_grid)[3]
	
	comm.Barrier()
	
	for r_n in range(number_rank) : 
	    if r_n == rank and rank != 0 : 
		comm.Send([result_n[0],MPI.DOUBLE],dest=0,tag=1)
		comm.Send([result_n[1],MPI.DOUBLE],dest=0,tag=2)
		comm.Send([result_n[2],MPI.DOUBLE],dest=0,tag=3)
		if Tracer == True : 
		    comm.Send([result_n[3],MPI.DOUBLE],dest=0,tag=4)
		if Cloudy == True : 
		    comm.Send([result_n[3+m_m],MPI.DOUBLE],dest=0,tag=5)
		if Composition == True :
		    comm.Send([result_n[3+m_m+c_c],MPI.DOUBLE],dest=0,tag=6)
	    elif r_n != 0 and rank == 0 : 
	        n_lay_rank_ne = repartition(n_layers,number_rank,r_n,False)
		if r_n == n_extra : 
		    for i_extra in range(extra) : 
	                n_lay_rank_ne = np.append(n_lay_rank_ne,n_layers+i_extra)
		result_n_P = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
		comm.Recv([result_n_P,MPI.DOUBLE],source=r_n,tag=1)
		result_P[n_lay_rank_ne,:,:] = result_n_P
		result_n_T = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
		comm.Recv([result_n_T,MPI.DOUBLE],source=r_n,tag=2)
		result_T[n_lay_rank_ne,:,:] = result_n_T
		result_n_Cn = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
		comm.Recv([result_n_Cn,MPI.DOUBLE],source=r_n,tag=3)
		result_Cn[n_lay_rank_ne,:,:] = result_n_Cn
		if Tracer == True : 
		    result_n_Q = np.zeros((n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
		    comm.Recv([result_n_Q,MPI.DOUBLE],source=r_n,tag=4)
		    result_Q[n_lay_rank_ne,:,:] = result_n_Q
		if Cloudy == True : 
		    result_n_gen = np.zeros((c_species.size,n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
		    comm.Recv([result_n_gen,MPI.DOUBLE],source=r_n,tag=5)
		    result_gen[:,n_lay_rank_ne,:,:] = result_n_gen
		if Composition == True : 
		    result_n_comp = np.zeros((n_species.size+1,n_lay_rank_ne.size,theta_number,length),dtype=np.float64)
		    comm.Recv([result_n_comp,MPI.DOUBLE],source=r_n,tag=6)
		    result_comp[:,n_lay_rank_ne,:,:] = result_n_comp
		    
	if rank == 0 : 	
	
	    np.save("%s%s/%s/%s_P_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                t_selec,r_step,phi_rot,phi_obli),result_P)
	    del result_P,result_n_P
            np.save("%s%s/%s/%s_T_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                t_selec,r_step,phi_rot,phi_obli),result_T)
	    del result_T,result_n_T
	    np.save("%s%s/%s/%s_Q_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
	        t_selec,r_step,phi_rot,phi_obli),result_Cn)
	    del result_Cn,result_n_Cn

            if Tracer == True :
                np.save("%s%s/%s/%s_Cn_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                    result_Q)
		del result_Q,result_n_Q
            if Cloudy == True :
                np.save("%s%s/%s/%s_gen_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                    result_gen)
		del result_gen,result_n_gen
            if Composition == True :
                np.save("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                    result_comp)
		del result_comp,result_n_comp

        del result_n,order_grid


########################################################################################################################

    if Convert == True :
	
	if rank == 0 : 
	
	    P = np.load("%s%s/%s/%s_P_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,\
                reso_alt,t_selec,r_step,phi_rot,phi_obli))
            T = np.load("%s%s/%s/%s_T_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,\
                reso_alt,t_selec,r_step,phi_rot,phi_obli))
            if Tracer == True :
                Q = np.load("%s%s/%s/%s_Q_%i%i%i_%i_%i_%.2f_%.2f.npy"
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli))
            else :
                Q = np.array([])
            if Cloudy == True :
                gen = np.load("%s%s/%s/%s_gen_%i%i%i_%i_%i_%.2f_%.2f.npy"
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli))
            else :
                gen = np.array([])
            comp = np.load("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli))
	

########################################################################################################################

    if Convert == True :

        direc = "%s/%s/"%(name_file,opac_file)
	
	facto = 2
	n_rmd = number_rank*facto
	
	if rank == 0 : 
	
	    P_rmd, T_rmd, Q_rmd, gen_cond_rmd, composit_rmd, wher, indices, liste = sort_set_param(P,T,Q,gen,comp,Tracer,Cloudy,Composition)

    	    p = np.log10(P_rmd)
    	    p_min = np.amin(p)
    	    p_max = np.amax(p)
	    step_rmd = P_rmd.size/n_rmd
    	    rmind = np.zeros((2,n_rmd+1),dtype=np.float64)
    	    rmind[0,0] = 0

    	    for i in xrange(n_rmd) :
                if i != n_rmd-1 : 
		    rmind[0,i+1] = (i+1)*step_rmd
		else : 
		    rmind[0,i+1] = P_rmd.size
                rmind[1,i] = p[np.int(rmind[0,i])]
            rmind[1,i+1] = p_max
	    
	    convertator_save(P_rmd,T_rmd,rmind,Q_rmd,gen_cond_rmd,composit_rmd,path,direc,reso_long,reso_lat,name_exo,t,\
                    x_step,phi_rot,phi_obli,domain,dim_bande,dim_gauss,Kcorr,Tracer,Cloudy,Composition)
	    
	    del P,T,Q,gen,comp,P_rmd,T_rmd,Q_rmd,gen_cond_rmd,composit_rmd,rmind
  	
	rank_ref = 0
	
	if rank != 0 : 
	    i_p_deb = facto*rank
	else : 
	    i_p_deb = 0
	if rank != number_rank-1 : 
	    i_p_fin = facto*(rank+1)+1
	else : 
	    i_p_fin = n_rmd+1
	
	comm.Barrier()	 
	
########################################################################################################################  
	
        if Kcorr == True :
            
	    rmind = np.load("%s%s/%s/rmind_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
            rmind = np.array(rmind[:,i_p_deb:i_p_fin],dtype=np.int)
	    T_rmd = np.load("%s%s/%s/T_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
	    T_rmd = T_rmd[rmind[0,0]:rmind[0,rmind[0].size-1]]
            P_rmd = np.load("%s%s/%s/P_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
	    P_rmd = P_rmd[rmind[0,0]:rmind[0,rmind[0].size-1]]
	    composit_rmd = np.load("%s%s/%s/compo_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
	    composit_rmd = composit_rmd[:,rmind[0,0]:rmind[0,rmind[0].size-1]]
            if Cl == True : 
	        gen_rmd = np.load("%s%s/%s/gen_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
		gen_rmd = gen_rmd[:,rmind[0,0]:rmind[0,rmind[0].size-1]]
  	    else : 
	        gen_rmd = np.array([])
            if Tracer == True :
                Q_rmd = np.load("%s%s/%s/Q_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
		Q_rmd = Q_rmd[rmind[0,0]:rmind[0,rmind[0].size-1]]
            else :
                Q_rmd = np.array([])
            
        else :
	    rmind = np.load("%s%s/%s/rmind_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	    rmind = np.array(rmind[:,i_p_deb:i_p_fin],dtype=np.int)
            T_rmd = np.load("%s%s/%s/T_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	    T_rmd = T_rmd[rmind[0,0]:rmind[0,rmind[0].size-1]]
            P_rmd = np.load("%s%s/%s/P_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	    P_rmd = P_rmd[rmind[0,0]:rmind[0,rmind[0].size-1]]
	    composit_rmd = np.load("%s%s/%s/compo_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	    composit_rmd = composit_rmd[:,rmind[0,0]:rmind[0,rmind[0].size-1]]
            if Cl : 
	        gen_rmd = np.load("%s%s/%s/gen_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
		gen_rmd = gen_rmd[:,rmind[0,0]:rmind[0,rmind[0].size-1]]
	    else : 
	        gen_rmd = np.array([])
            if Tracer == True :
                Q_rmd = np.load("%s%s/%s/Q_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
		Q_rmd = Q_rmd[rmind[0,0]:rmind[0,rmind[0].size-1]]
            else :
                Q_rmd = np.array([])
	
	data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                reso_lat))
		

########################################################################################################################
		
        if Kcorr == True :
            gauss = np.arange(0,dim_gauss,1)
            gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
            P_sample = np.load("%s%s/P_sample.npy"%(path,name_source))
	    '''
	    wh_p, = np.where((np.log10(P_sample)>=P_rmd[0])*(np.log10(P_sample)>=P_rmd[P_rmd.size-1]))
	    if wh_p.size == 0 : 
	        wh_p = np.array([0,1])
	    elif wh_p.size == 1 : 
	        if wh_p[0] != P_sample.size-1 : 
		    wh_p = np.array([wh_p[0],wh_p[0]+1])
		else : 
		    wh_p = np.array([wh_p[0]-1,wh_p[0]])
	    P_sample = P_sample[wh_p]
	    '''
            T_sample = np.load("%s%s/T_sample.npy"%(path,name_source))
            if Tracer == True :
                Q_sample = np.load("%s%s/Q_sample.npy"%(path,name_source))
            else :
                Q_sample = np.array([])
            bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,domain))

            k_corr_data_grid = "%s%s/k_corr_%s_%s.npy"%(path,name_source,name_exo,domain)
        else :
            gauss = np.array([])
            gauss_val = np.array([])
            P_sample = np.load("%s%s/P_sample_%s.npy"%(path,name_source,source))
	    '''
	    wh_p, = np.where((np.log10(P_sample)>=P_rmd[0])*(np.log10(P_sample)>=P_rmd[P_rmd.size-1]))
	    if wh_p.size == 0 : 
	        wh_p = np.array([0,1])
	    elif wh_p.size == 1 : 
	        if wh_p[0] != P_sample.size-1 : 
		    wh_p = np.array([wh_p[0],wh_p[0]+1])
		else : 
		    wh_p = np.array([wh_p[0]-1,wh_p[0]])
            '''
            T_sample = np.load("%s%s/T_sample_%s.npy"%(path,name_source,source))
            if Tracer == True :
                Q_sample = np.load("%s%s/Q_sample_%s.npy"%(path,name_source,source))
            else :
                Q_sample = np.array([])
            bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

            k_corr_data_grid = "%s%s/crossection_%s.npy"%(path,name_source,source)

        # Telechargement des donnees CIA

        if Cont == True :
            k_cont_h2h2 = "%s%s/k_cont_h2h2.npy"%(path,name_source)
            k_cont_h2he = "%s%s/k_cont_h2he.npy"%(path,name_source)
            k_cont_nu = np.load("%s%s/K_cont_nu_h2h2.npy"%(path,name_source))
            T_cont = np.load("%s%s/T_cont_h2h2.npy"%(path,name_source))
        else :
            k_cont_h2h2 = np.array([])
            k_cont_h2he = np.array([])
            k_cont_nu = np.array([])
            T_cont = np.array([])

        # Telechargement des donnees nuages

        if Cl == True :
            bande_cloud = np.load("%s%s/bande_cloud_%s.npy"%(path,name_source,name_exo))
            r_cloud = np.load("%s%s/radius_cloud_%s.npy"%(path,name_source,name_exo))
            cl_name = ''
            for i in range(c_species_name.size) :
                cl_name += '%s_'%(c_species_name[i])
            Q_cloud = "%s%s/Q_%s%s.npy"%(path,name_source,cl_name,name_exo)
            message_clouds = ''
            for i in range(c_species.size) :
                message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff*10**6,rho_p[i]/1000.)
        else :
            bande_cloud = np.array([])
            r_cloud = np.array([])
            Q_cloud = np.array([])


########################################################################################################################

        wh_p = 0
	
	convertator (P_rmd,T_rmd,gen_rmd,c_species,Q_rmd,composit_rmd,ind_active,k_corr_data_grid,k_cont_h2h2,k_cont_h2he,k_cont_nu,k_cont_nu,\
                     Q_cloud,T_cont,P_sample,T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,direc,\
                     t,phi_rot,phi_obli,n_species,domain,ratio,path,name_exo,reso_long,reso_lat,wh_p,rank,rank_ref,\
                     Tracer,Molecular,Cont,Cl,Composition,Scatt,Kcorr,Optimal)

########################################################################################################################


comm.Barrier()


########################################################################################################################
########################################################################################################################
##########################################      TRANSFERT 3D      ######################################################
########################################################################################################################
########################################################################################################################

if Cylindric_transfert_3D == True :

    if rank == 0 : 
        print('Download of stiches array')
    
    order_grid = np.load("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
    if Module == True :
        z_grid = np.load("%s%s/%s/z_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
    else :
        z_grid = np.array([])

    if Discreet == True :
        dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        pdx_grid = np.array([])

    else :
    
        pdx_grid = np.load("%s%s/%s/pdx_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                       %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                      %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))

    data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                reso_lat))

########################################################################################################################

    if rank == 0 : 
        print('Download of couples array')

    if Kcorr == True :
        T_rmd = np.load("%s%s/%s/T_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        P_rmd = np.load("%s%s/%s/P_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        gen_rmd = np.load("%s%s/%s/gen_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        if Tracer == True :
            Q_rmd = np.load("%s%s/%s/Q_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        else :
            Q_rmd = np.array([])
        rmind = np.load("%s%s/%s/rmind_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
    else :
        T_rmd = np.load("%s%s/%s/T_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        P_rmd = np.load("%s%s/%s/P_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        gen_rmd = np.load("%s%s/%s/gen_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        if Tracer == True :
            Q_rmd = np.load("%s%s/%s/Q_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        else :
            Q_rmd = np.array([])
        rmind = np.load("%s%s/%s/rmind_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))    

########################################################################################################################
    
    if rank == 0 : 
        print 'Download of opacities data'
    
    rank_ref = 0
    
    # Le simulateur de spectre va decouper en bande de facon a limiter au maximum la saturation en memoire
    # Une option permet un decoupage supplementaire en the ta ou exclusivement en theta si les tableaux de donnees ne 
    # sont pas trop importants.
    
    dom_rank = repartition(dim_bande,number_rank,rank,True)
    
    cases = np.zeros(4,dtype=np.int)
    cases_names = ['molecular','continuum','scattering','clouds']
    if Isolated == False : 
        cases[0] = 1
	if Continuum == True : 
	    cases[1] = 1
	if Scattering == True :
	    cases[2] = 1
	if Clouds == True :
	    cases[3] = 1 
    else : 
        cases[0] = 0
        if Continuum == True : 
	    cases[1] = 1
	if Scattering == True :
	    cases[2] = 1
	if Clouds == True :
	    cases[3] = 1
    
    wh_ca, = np.where(cases == 1)
    
    for i_ca in range(wh_ca.size) : 
        
	proc = np.array([True,False,False,False])
	if wh_ca[i_ca] != 0 : 
	    proc[wh_ca[i_ca]] = True
	else : 
	    proc[wh_ca[i_ca]] = False
	Isolated, Continuum, Scattering, Clouds = proc[0],proc[1],proc[2],proc[3]
	
	if rank == 0 :
	    stud = stud_type(r_eff,Single,Continuum,Isolated,Scattering,Clouds)
	    save_name_3D_step = saving('3D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
                phi_rot,r_eff,domain,stud,lim_alt,rupt_alt,Discreet,Integration,Module,Optimal,Kcorr)	    

        if Isolated == False :
            if Kcorr == True :
                k_rmd = np.load("%s%s/%s/k_corr_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
	        k_rmd = k_rmd[:,dom_rank,:]
                gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
            else :
                if Optimal == True :
                    k_rmd = np.load("%s%s/%s/k_cross_opt_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
		    k_rmd = k_rmd[:,dom_rank]
                else :
                    k_rmd = np.load("%s%s/%s/k_cross_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                    %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
		    k_rmd = k_rmd[:,dom_rank]
                gauss_val = np.array([])
        else :
            if Kcorr == True :
                k_rmd = np.load("%s%s/%s/k_corr_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
	        k_rmd = k_rmd[:,dom_rank,:]
                k_rmd = np.shape(k_rmd)
            else :
                k_rmd = np.load("%s%s/%s/k_cross_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	        k_rmd = k_rmd[:,dom_rank]
                k_rmd = np.shape(k_rmd)
            gauss_val = np.array([])

        if Continuum == True :
            if Kcorr == True :
                k_cont_rmd = np.load("%s%s/%s/k_cont_h2heFS_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
	        k_cont_rmd = np.transpose(k_cont_rmd[dom_rank,:])
            else :
                k_cont_rmd = np.load("%s%s/%s/k_cont_h2heFS_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	        k_cont_rmd = np.transpose(k_cont_rmd[dom_rank,:])
        else :
            k_cont_rmd = np.array([])

        if Scattering == True :
            if Kcorr == True :
                k_sca_rmd = np.load("%s%s/%s/k_sca_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
	        k_sca_rmd = k_sca_rmd[:,dom_rank]
            else :
                k_sca_rmd = np.load("%s%s/%s/k_sca_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
	        k_sca_rmd = k_sca_rmd[:,dom_rank]
        else :
            k_sca_rmd = np.array([])

        if Clouds == True :
            if Kcorr == True :
                k_cloud_rmd = np.load("%s%s/%s/k_cloud_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%.2f_%s.npy" \
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                r_eff*10**6,domain))
	        k_cloud_rmd = k_cloud_rmd[:,:,dom_rank]
            else :
                k_cloud_rmd = np.load("%s%s/%s/k_cloud_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%.2f_%s.npy" \
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,r_eff*10**6,domain))
	        k_cloud_rmd = k_cloud_rmd[:,:,dom_rank]
        else :
            k_cloud_rmd = np.array([])

########################################################################################################################
    
        if rank == 0 : 
	    print 'Pytmosph3R will begin to compute the %s contribution'%(cases_names[wh_ca[i_ca]])
	
	I_n = trans2fert3D (k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,Rp,h,g0,r_step,theta_step,gauss_val,dom_rank.size,data_convert,\
                  P_rmd,T_rmd,Q_rmd,dx_grid,order_grid,pdx_grid,z_grid,t,\
                  name_file,n_species,Single,rmind,lim_alt,rupt_alt,rank,rank_ref,\
                  Tracer,Continuum,Isolated,Scattering,Clouds,Kcorr,Rupt,Module,Integration,TimeSel)
    
        if rank == 0 : 
            sh_I = np.shape(I_n)
	    r_size, theta_size = sh_I[1], sh_I[2]
	    Itot = np.zeros((dim_bande,r_size,theta_size),dtype=np.float64)
            Itot[dom_rank,:,:] = I_n
        else : 
            I_n = np.array(I_n,dtype=np.float64)
	    comm.Send([I_n,MPI.DOUBLE],dest=0,tag=0)

        if rank == 0 : 
            for i_n in range(1,number_rank) : 
	        new_dom_rank = repartition(dim_bande,number_rank,r_n,True)
	        I_rn = np.zeros((new_dom_rank.size,r_size,theta_size),dtype=np.float64)
	        comm.Recv([I_rn,MPI.DOUBLE],source=i_n,tag=0)
	        Itot[new_dom_rank,:,:] = I_rn
	        print i_n
	
	if rank == 0 :
	    np.save(save_name_3D_step,Itot)
	    del Itot
	del I_n   
    
    if rank == 0 :
        tau = 0
	for i_ca in range(wh_ca.size) : 
	    proc = np.array([False,False,False,False])
	    proc[wh_ca[i_ca]] = True
	    Isolated, Continuum, Scattering, Clouds = proc[0],proc[1],proc[2],proc[3]
	    stud = stud_type(r_eff,Single,Continuum,Isolated,Scattering,Clouds)
	    save_name_3D_step = saving('3D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
                phi_rot,r_eff,domain,stud,lim_alt,rupt_alt,Discreet,Integration,Module,Optimal,Kcorr)
	    I_step = np.load(save_name_3D_step)
	    tau += np.log(I_step)
	Itot = np.exp(I_tot)
        np.save(save_name_3D,Itot)

########################################################################################################################

if View == True :

    if Cylindric_transfert_3D == False :
        Itot = np.load(save_name_3D)
    if Kcorr == True :
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,domain))
    else :
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(Itot,bande_sample,Rs,Rp,r_step,0,\
                                                                                False,Kcorr,Middle)

    if Radius == True :
        plt.semilogx()
        plt.grid(True)
        plt.plot(1/(100.*bande_sample)*10**6,R_eff,'g',linewidth = 2,label='3D spectrum')
        plt.ylabel('Effective radius (m)')
        plt.xlabel('Wavelenght (micron)')
        plt.legend(loc=4)
        plt.show()

    if Flux == True :
        plt.semilogx()
        plt.grid(True)
        plt.plot(1/(100.*bande_sample)*10**6,flux,'r',linewidth = 2,label='3D spectrum')
        plt.ylabel('Flux (Rp/Rs)2')
        plt.xlabel('Wavelenght (micron)')
        plt.legend(loc=4)
        plt.show()

########################################################################################################################


if rank == 0 : 
    print 'Pytmosph3R process finished with success'
