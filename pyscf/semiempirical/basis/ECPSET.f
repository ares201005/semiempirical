      SUBROUTINE ECPSET (ES,CS,CP,NGS,NI)
C     *
C     ENERGY-OPTIMIZED VALENCE BASIS SETS FOR EFFECTIVE CORE POTENTIALS.
C     *
C     ORIGINAL 4G BASIS SETS FOR THE ELEMENTS Li-Ar FROM:
C     W.J.STEVENS, H.BASCH, AND M.KRAUSS, J.CHEM.PHYS. 81, 6026 (1984).
C     FIT TO 3G BASIS SETS FOR THE ELEMENTS Li-Ne, SEE:
C     M.KOLB AND W.THIEL, J.COMPUT.CHEM. 14, 775 (1993) - TABLE I.
C     THIS ROUTINE PROVIDES BOTH THE 3G AND 4G SETS (OPTIONALLY).
C     *
C     NOTATION. I=INPUT, O=OUTPUT.
C     ES()      EXPONENTS OF PRIMITIVES (O). SHARED BETWEEN S AND P.
C     CS()      S CONTRACTION COEFFICIENTS (O).
C     CP()      P CONTRACTION COEFFICIENTS (O).
C     NGS       CONTRACTION LENGTH (I,O).
C               THE VALUE OF NGS IS NORMALLY DEFINED BY DEFAULT (O).
C               NGS INPUT IS ONLY RELEVANT TO SELECT 4G SET FOR Li-Ne.
C     NI        ATOMIC NUMBER (I).
C     *
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON /NBFILE/ NBF(20)
      DIMENSION ES(4),CS(4),CP(4)
C *** BRANCH FOR ELEMENT WITH ATOMIC NUMBER NI.
C     TREAT QM/MM CONNECTION ATOM (NI=86) LIKE A CARBON ATOM.
      IF(NI.NE.86) THEN
         IF(NI.LE.2 .OR. NI.GT.18) GO TO 400
         IF(NI.GT.10) GO TO 200
         IF(NGS.EQ.4) GO TO 100
         NGS = 3
         IGO = NI-2
      ELSE
         NGS = 3
         IGO = 4
      ENDIF
      GO TO (10,20,30,40,50,60,70,80),IGO
C *** LITHIUM
   10 ES(1)  =  0.44227D+00
      ES(2)  =  0.07847D+00
      ES(3)  =  0.02434D+00
      CS(1)  = -0.18805D+00
      CS(2)  =  0.66552D+00
      CS(3)  =  0.47762D+00
      CP(1)  =  0.11671D+00
      CP(2)  =  0.52628D+00
      CP(3)  =  0.52282D+00
      RETURN
C *** BERYLLIUM
   20 ES(1)  =  1.06039D+00
      ES(2)  =  0.20758D+00
      ES(3)  =  0.05933D+00
      CS(1)  = -0.18525D+00
      CS(2)  =  0.52418D+00
      CS(3)  =  0.62320D+00
      CP(1)  =  0.15121D+00
      CP(2)  =  0.55940D+00
      CP(3)  =  0.47765D+00
      RETURN
C *** BORON
   30 ES(1)  =  1.72427D+00
      ES(2)  =  0.35009D+00
      ES(3)  =  0.09394D+00
      CS(1)  = -0.19359D+00
      CS(2)  =  0.61497D+00
      CS(3)  =  0.54976D+00
      CP(1)  =  0.18292D+00
      CP(2)  =  0.54437D+00
      CP(3)  =  0.48303D+00
      RETURN
C *** CARBON
   40 ES(1)  =  2.64486D+00
      ES(2)  =  0.54215D+00
      ES(3)  =  0.14466D+00
      CS(1)  = -0.19188D+00
      CS(2)  =  0.61628D+00
      CS(3)  =  0.54896D+00
      CP(1)  =  0.20259D+00
      CP(2)  =  0.55830D+00
      CP(3)  =  0.45514D+00
      RETURN
C *** NITROGEN
   50 ES(1)  =  3.68849D+00
      ES(2)  =  0.77534D+00
      ES(3)  =  0.20498D+00
      CS(1)  = -0.19269D+00
      CS(2)  =  0.61888D+00
      CS(3)  =  0.54926D+00
      CP(1)  =  0.22281D+00
      CP(2)  =  0.56032D+00
      CP(3)  =  0.43859D+00
      RETURN
C *** OXYGEN
   60 ES(1)  =  4.78499D+00
      ES(2)  =  0.99860D+00
      ES(3)  =  0.25687D+00
      CS(1)  = -0.19248D+00
      CS(2)  =  0.66952D+00
      CS(3)  =  0.50270D+00
      CP(1)  =  0.24158D+00
      CP(2)  =  0.55890D+00
      CP(3)  =  0.43160D+00
      RETURN
C *** FLUORINE
   70 ES(1)  =  6.01783D+00
      ES(2)  =  1.25315D+00
      ES(3)  =  0.31760D+00
      CS(1)  = -0.18850D+00
      CS(2)  =  0.69800D+00
      CS(3)  =  0.47427D+00
      CP(1)  =  0.25667D+00
      CP(2)  =  0.56013D+00
      CP(3)  =  0.42139D+00
      RETURN
C *** NEON
   80 ES(1)  =  7.47831D+00
      ES(2)  =  1.55488D+00
      ES(3)  =  0.39057D+00
      CS(1)  = -0.18692D+00
      CS(2)  =  0.70988D+00
      CS(3)  =  0.46248D+00
      CP(1)  =  0.26586D+00
      CP(2)  =  0.56134D+00
      CP(3)  =  0.41439D+00
      RETURN
C *** FIRST-ROW ATOMS, ORIGINAL 4G EXPANSIONS (STEVENS ET AL).
  100 IGO    = NI-2
      NGS    = 4
      GO TO (110,120,130,140,150,160,170,180),IGO
C *** LITHIUM
  110 ES(1)  =  0.61770D+00
      ES(2)  =  0.14340D+00
      ES(3)  =  0.05048D+00
      ES(4)  =  0.01923D+00
      CS(1)  = -0.16287D+00
      CS(2)  =  0.12643D+00
      CS(3)  =  0.76179D+00
      CS(4)  =  0.21800D+00
      CP(1)  =  0.06205D+00
      CP(2)  =  0.24719D+00
      CP(3)  =  0.52140D+00
      CP(4)  =  0.34290D+00
      RETURN
C *** BERYLLIUM
  120 ES(1)  =  1.44700D+00
      ES(2)  =  0.35220D+00
      ES(3)  =  0.12190D+00
      ES(4)  =  0.04395D+00
      CS(1)  = -0.15647D+00
      CS(2)  =  0.10919D+00
      CS(3)  =  0.67538D+00
      CS(4)  =  0.32987D+00
      CP(1)  =  0.08924D+00
      CP(2)  =  0.30999D+00
      CP(3)  =  0.51812D+00
      CP(4)  =  0.27911D+00
      RETURN
C *** BORON
  130 ES(1)  =  2.71000D+00
      ES(2)  =  0.65520D+00
      ES(3)  =  0.22480D+00
      ES(4)  =  0.07584D+00
      CS(1)  = -0.14987D+00
      CS(2)  =  0.08442D+00
      CS(3)  =  0.69751D+00
      CS(4)  =  0.32842D+00
      CP(1)  =  0.09474D+00
      CP(2)  =  0.30807D+00
      CP(3)  =  0.46876D+00
      CP(4)  =  0.35025D+00
      RETURN
C *** CARBON
  140 ES(1)  =  4.28600D+00
      ES(2)  =  1.04600D+00
      ES(3)  =  0.34470D+00
      ES(4)  =  0.11280D+00
      CS(1)  = -0.14722D+00
      CS(2)  =  0.08125D+00
      CS(3)  =  0.71360D+00
      CS(4)  =  0.31521D+00
      CP(1)  =  0.10257D+00
      CP(2)  =  0.32987D+00
      CP(3)  =  0.48212D+00
      CP(4)  =  0.31593D+00
      RETURN
C *** NITROGEN
  150 ES(1)  =  6.40300D+00
      ES(2)  =  1.58000D+00
      ES(3)  =  0.50940D+00
      ES(4)  =  0.16230D+00
      CS(1)  = -0.13955D+00
      CS(2)  =  0.05492D+00
      CS(3)  =  0.71678D+00
      CS(4)  =  0.33210D+00
      CP(1)  =  0.10336D+00
      CP(2)  =  0.33205D+00
      CP(3)  =  0.48708D+00
      CP(4)  =  0.31312D+00
      RETURN
C *** OXYGEN
  160 ES(1)  =  8.51900D+00
      ES(2)  =  2.07300D+00
      ES(3)  =  0.64710D+00
      ES(4)  =  0.20000D+00
      CS(1)  = -0.14551D+00
      CS(2)  =  0.08286D+00
      CS(3)  =  0.74325D+00
      CS(4)  =  0.28472D+00
      CP(1)  =  0.11007D+00
      CP(2)  =  0.34969D+00
      CP(3)  =  0.48093D+00
      CP(4)  =  0.30727D+00
      RETURN
C *** FLUORINE
  170 ES(1)  = 11.12000D+00
      ES(2)  =  2.68700D+00
      ES(3)  =  0.82100D+00
      ES(4)  =  0.24750D+00
      CS(1)  = -0.14451D+00
      CS(2)  =  0.08971D+00
      CS(3)  =  0.75659D+00
      CS(4)  =  0.26570D+00
      CP(1)  =  0.11300D+00
      CP(2)  =  0.35841D+00
      CP(3)  =  0.48002D+00
      CP(4)  =  0.30381D+00
      RETURN
C *** NEON
  180 ES(1)  = 14.07000D+00
      ES(2)  =  3.38900D+00
      ES(3)  =  1.02100D+00
      ES(4)  =  0.30310D+00
      CS(1)  = -0.14463D+00
      CS(2)  =  0.09331D+00
      CS(3)  =  0.76297D+00
      CS(4)  =  0.25661D+00
      CP(1)  =  0.11514D+00
      CP(2)  =  0.36479D+00
      CP(3)  =  0.48052D+00
      CP(4)  =  0.29896D+00
      RETURN
C *** SECOND-ROW ATOMS, ORIGINAL 4G EXPANSIONS (STEVENS ET AL).
  200 IGO    = NI-10
      NGS    = 4
      GO TO (210,220,230,240,250,260,270,280),IGO
C *** SODIUM
  210 ES(1)  =  0.42990D+00
      ES(2)  =  0.08897D+00
      ES(3)  =  0.03550D+00
      ES(4)  =  0.01455D+00
      CS(1)  = -0.20874D+00
      CS(2)  =  0.31206D+00
      CS(3)  =  0.70300D+00
      CS(4)  =  0.11648D+00
      CP(1)  = -0.02571D+00
      CP(2)  =  0.21608D+00
      CP(3)  =  0.54196D+00
      CP(4)  =  0.35484D+00
      RETURN
C *** MAGNESIUM
  220 ES(1)  =  0.66060D+00
      ES(2)  =  0.18450D+00
      ES(3)  =  0.06983D+00
      ES(4)  =  0.02740D+00
      CS(1)  = -0.24451D+00
      CS(2)  =  0.25323D+00
      CS(3)  =  0.69720D+00
      CS(4)  =  0.21655D+00
      CP(1)  = -0.04421D+00
      CP(2)  =  0.27323D+00
      CP(3)  =  0.57626D+00
      CP(4)  =  0.28152D+00
      RETURN
C *** ALUMINIUM
  230 ES(1)  =  0.90110D+00
      ES(2)  =  0.44950D+00
      ES(3)  =  0.14050D+00
      ES(4)  =  0.04874D+00
      CS(1)  = -0.30377D+00
      CS(2)  =  0.13382D+00
      CS(3)  =  0.76037D+00
      CS(4)  =  0.32232D+00
      CP(1)  = -0.07929D+00
      CP(2)  =  0.16540D+00
      CP(3)  =  0.53015D+00
      CP(4)  =  0.47724D+00
      RETURN
C *** SILICIUM
  240 ES(1)  =  1.16700D+00
      ES(2)  =  0.52680D+00
      ES(3)  =  0.18070D+00
      ES(4)  =  0.06480D+00
      CS(1)  = -0.32403D+00
      CS(2)  =  0.18438D+00
      CS(3)  =  0.77737D+00
      CS(4)  =  0.26767D+00
      CP(1)  = -0.08450D+00
      CP(2)  =  0.23786D+00
      CP(3)  =  0.56532D+00
      CP(4)  =  0.37433D+00
      RETURN
C *** PHOSPHORUS
  250 ES(1)  =  1.45900D+00
      ES(2)  =  0.65490D+00
      ES(3)  =  0.22560D+00
      ES(4)  =  0.08115D+00
      CS(1)  = -0.34091D+00
      CS(2)  =  0.21535D+00
      CS(3)  =  0.79578D+00
      CS(4)  =  0.23092D+00
      CP(1)  = -0.09378D+00
      CP(2)  =  0.29205D+00
      CP(3)  =  0.58688D+00
      CP(4)  =  0.30631D+00
      RETURN
C *** SULFUR
  260 ES(1)  =  1.81700D+00
      ES(2)  =  0.83790D+00
      ES(3)  =  0.28540D+00
      ES(4)  =  0.09939D+00
      CS(1)  = -0.34015D+00
      CS(2)  =  0.19601D+00
      CS(3)  =  0.82666D+00
      CS(4)  =  0.21652D+00
      CP(1)  = -0.10096D+00
      CP(2)  =  0.31244D+00
      CP(3)  =  0.57906D+00
      CP(4)  =  0.30748D+00
      RETURN
C *** CHLORINE
  270 ES(1)  =  2.22500D+00
      ES(2)  =  1.17300D+00
      ES(3)  =  0.38510D+00
      ES(4)  =  0.13010D+00
      CS(1)  = -0.33098D+00
      CS(2)  =  0.11528D+00
      CS(3)  =  0.84717D+00
      CS(4)  =  0.26534D+00
      CP(1)  = -0.12604D+00
      CP(2)  =  0.29952D+00
      CP(3)  =  0.58357D+00
      CP(4)  =  0.34097D+00
      RETURN
C *** ARGON
  280 ES(1)  = 2.70600D+00
      ES(2)  =  1.27800D+00
      ES(3)  =  0.43540D+00
      ES(4)  =  0.14760D+00
      CS(1)  = -0.31286D+00
      CS(2)  =  0.11821D+00
      CS(3)  =  0.86786D+00
      CS(4)  =  0.22264D+00
      CP(1)  = -0.10927D+00
      CP(2)  =  0.32601D+00
      CP(3)  =  0.57952D+00
      CP(4)  =  0.30349D+00
      RETURN
C *** ERROR EXIT FOR ELEMENTS WITHOUT ECP BASIS
  400 CONTINUE
      NB6 = NBF(6)
      WRITE(NB6,500) NI
      STOP 'ECPSET'
  500 FORMAT(/1X,'INPUT ERROR IN SUBROUTINE ECPSET.',
     1        1X,'NO ECP BASIS AVAILABLE FOR ELEMENT',I3/)
      END
