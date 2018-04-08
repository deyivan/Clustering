%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  P r � c t i c a   B l o q u e  1 �  ( C l u s t e r i n g )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  A u t o r :   J o s �   L u i s   M u � o z   A m p u e r o
% D e s a r r o l l o   d e   S i s t .   I n t e l i g e n t e s
% M � s t e r   e n   I n g e n i e r � a   I n f o r m � t i c a
%         ( E S I - C i u d a d   R e a l ,   U C L M )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, close all, clc;              % Limpieza
load('./Renta_local_2007.mat','T'); % carga de datos (formato MATLAB)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% S e l e c c i � n   d e   C a r � c t e r � s t i c a s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X=  [];
X=  [T{:,'Codigo_CA'},...
     T{:,'Rimp_Agregada'},T{:,'Rimp_Declarante'},T{:,'Rimp_Habitante'},...
     T{:,'DRL_Q1'},T{:,'DRL_Q2'},T{:,'DRL_Q3'},T{:,'DRL_Q4'},T{:,'DRL_Q5'},...
     T{:,'CRL_Top_1'},T{:,'CRL_Top_05'},T{:,'CRL_Top_01'}];

varX= size(X,1); % N�mero de observaciones: Ciudades
dimX= size(X,2); % N�mero de dimensiones:   Caracter�sticas
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          E s t a n d a r i z a c i � n   d e   l a s
%  o b s e r v a c i o n e s  ( m � t o d o    z - s c o r e )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:dimX
    X(:,i)= (X(:,i)-mean(X(:,i)))/std(X(:,i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C � l c u l o   d e l   B I C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Kmax= 10;   % N�mero m�ximo de agrupaciones
minK= Kmax; % Se inicializa el n�mero de grupos que hacen m�nimo K al m�ximo
cidxK= [];  % Se mantiene una estructura para guardar la asignaci�n a cada
            % grupo en cada evaluaci�n del BIC y se inicializa con 0's
cidxK(1:varX,:)= 0;

% Se evaluar� el BIC para cada cluster del intervalo [2..Kmax]
for K=2:Kmax
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % E j e c u c i � n   a l g o r i t m o    d e    l a s 
    %                      K - m e d i a s 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
    % p a r � m e t r o s   d e   e n t r a d a
    % X matriz de datos, filas individuos, columnas atributos
    % K=minK numero de grupos a conformar (recomendado por BIC)
    % 'Replicates' n�mero de repeticiones, 50 en este caso 
    % 'Distance'  distancia usada, 'sqEuclidean'->el cuadrado de 
    %             la eucl�dea, tambi�n se puede elegir 'city'->L1
    %%%%%%%
    % p a r � m e t r o s   d e   s a l i d a              
    % cidx(i) devuelve el conglogmerado al que pertencece el dato i
    % ctrs    centroides de los grupos
    % sumd    suma de la distancia intracluster 
    % D       matriz de distancia de cada objeto (fila) a cada centroide (columna)
    %%%%%%%%   
    
    [cidx,ctrs,sumd,D]= kmeans(X, K,'Replicates',25,...
                        'Distance','sqEuclidean', 'MaxIter',100);
    cidxK= [cidxK,cidx];        % Almacenamos y conservamos la asignaci�n de
                                % cada observaci�n al cl�ster K
    [Bic_K,xi]= BIC(K,cidx,X);  % Llamada a la funci�n BIC    
    BICK(K)= Bic_K;             % Almacenamos el resultado del BIC para la�
                                % evaluaci�n de K,cidx y K
    % Si el BIC obtenido en la evaluaci�n del K actual es el menor (hasta
    % el momento) de la serie hist�rica [2..K]
    if Bic_K == min(BICK(2:size(BICK,2)))
        minK=K;                 % El grupo que hace m�nimo a Kmeans es K.
        % De este modo, no habr� que ejecutar el algoritmo a posteriori
        % para el menor K encontrado, con lo que solo habr� que extraerlo
        % del vector cidxK
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R e s u l t a d o   d e   l a   e v a l u a c i � n   d e l   
%  B I C   p a r a   e l   i n t e r v a l o   2 .. K m a x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
plot(2:K',BICK(2:K)','s-','MarkerSize',6,...
     'MarkerEdgeColor','r', 'MarkerFaceColor','r');
xlabel('K','fontsize',18);      % etiquetado del eje-x
ylabel('BIC(K)','fontsize',18); % etiquetado del eje-y

% Recuperamos la mejor asignaci�n obtenida del BIC
best_cdix=  cidxK(:,minK);
% A�adimos el cluster K al que pertenece cada ciudad en la tabla original
T.Group_ID = best_cdix(:,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  V a l i d a c i � n   d e   l a   e v a l u a c i � n   d e l   
% B I C   s e g � n   e l   c r i t e r i o   s i l h o u e t t e
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2);
silhouette(X,best_cdix(:,1),'sqEuclidean');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R e d u c c i � n   d e   l a   d i m e n s i o n a l i d a d
%  " P r i n c i p a l   C o m p o n e n t   A n a l y s i s "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[coeff,score,latent,tsquared,explained] = pca(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   A n � l i s i s   d e   l a s   c o m p o n e n t e s
% p r i n c i p a l e s   e n   d o s   d i m e n s i o n e s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3);
palette= colormap(hsv(minK));     % Se selecciona la paleta de colores
scatter(score(:,1),score(:,2),...
        20, palette(best_cdix(:,1),:), 'filled',...
        'MarkerEdgeColor','Black', 'LineWidth',0.01,'Marker','o');
xlabel('1� Componente Principal');
ylabel('2� Componente Principal');

% Leyenda din�mica en funci�n del minK obtenido
for K=1:minK
    p(K)= patch(NaN, NaN, palette(K,:));
    p(K).DisplayName= strcat('K=', int2str(K));
end
legend(p);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   A n � l i s i s   d e   l a s   c o m p o n e n t e s
% p r i n c i p a l e s   e n   t r e s   d i m e n s i o n e s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(4);
% Se invoca la misma paleta de colores para forzar la coincidencia de
% colores entre distintos gr�ficos
palette= colormap(hsv(minK));                        
scatter3(score(:,1),score(:,2),score(:,3),...
        20, palette(best_cdix(:,1),:), 'filled',...
        'MarkerEdgeColor','Black', 'LineWidth',0.01, 'Marker','o');
view(3), axis vis3d, box on, rotate3d on;   
xlabel('1� Componente Principal');
ylabel('2� Componente Principal');
zlabel('3� Componente Principal');

% Leyenda din�mica en funci�n del minK obtenido
for K=1:minK
    p(K)= patch(NaN, NaN, palette(K,:));
    p(K).DisplayName= strcat('K=', int2str(K));
end
legend(p);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   C a r a c t e r i z a c i � n   a n a l � t i c a   e n  
% b a s e   a   l o s   r e s u l t a d o s   o b t e n i d o s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v= [];
% Caracterizamos cada cada uno de los K-grupos con la informaci�n m�s
% significativa, de forma que su lectura sea lo m�s auto-contenida posible
for K=1:minK
    num_Instancias=     size(T(T.Group_ID==K,:),1);
    max_Rimp_agregada=  max(T(T.Group_ID==K,:).Rimp_Agregada);
    med_Rimp_agregada=  median(T(T.Group_ID==K,:).Rimp_Agregada);
    med_Rimp_declarante=median(T(T.Group_ID==K,:).Rimp_Declarante);
    med_Rimp_habitante= median(T(T.Group_ID==K,:).Rimp_Habitante);
    max_n_Habitantes=   max(T(T.Group_ID==K,:).N_Habitantes);
    med_n_Habitantes=   median(T(T.Group_ID==K,:).N_Habitantes);    
    med_I_Gini=         median(T(T.Group_ID==K,:).I_Gini);
    
    v= [v;K,num_Instancias,...      % A�adimos una fila por cada K-cluster
        max_Rimp_agregada,med_Rimp_agregada,...
        med_Rimp_declarante,med_Rimp_habitante,...
        max_n_Habitantes,med_n_Habitantes,med_I_Gini];
end
% S�lo a efectos de presentaci�n, se genera una tabla con la informaci�n
% contenida en el array 'v'
TgroupFeatures= array2table(v,'VariableNames',...
                {'Group_ID','Instances',...
                'max_Rimp_agregada','med_Rimp_agregada',...
                'med_Rimp_declarante','med_Rimp_habitante',...
                'max_n_Habitantes','med_n_Habitantes',...
                'med_I_Gini'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  V i s u a l i z a c i � n   d e   l o s   K - c l u s t e r  
%          e n   e l   m a p a   d e   E s p a � a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            

% Zonas geogr�ficas incluidas en Webmapping Toolboox
land= shaperead('landareas', 'UseGeoCoords', true);
% L�mites auton�micos importados desde el proyecto GADM (http://gadm.org/)
ccaa= shaperead('GADM_Spain/ESP_adm1.shp', 'UseGeoCoords', true);
oceanClr=   [.5 .7 .9];             % Color del oc�ano en el mapa
landClr=    [0.5 0.7 0.5];          % Color del terreno en el mapa
TfilterCA=  [];                     % Vector auxiliar para filtrar por CA

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M a p a   a u t o n � m i c o   d e  l a  p e n i n s u l a ,
% C e u t a ,   M e l i l l a   e   I s l a s   B a l e a r e s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(5)
palette= colormap(hsv(minK));     % Se reinicia la paleta de color 
ax= worldmap('Spain');              % Selecci�n del area geogr�fica
setm(ax, 'FFaceColor', oceanClr,... % Parametrizaci�n del mapa
    'MapProjection','mercator')     % Proyecci�n plana

% Se muestra el �rea geogr�fica y el mapa auton�mico
geoshow(ax, land, 'FaceColor', landClr)
geoshow(ccaa)
mlabel; plabel; gridm;              % Sin elementos auxiliares en el mapa
TfilterCA= T(T.Codigo_CA~=5,:);     % Filtramos todas las CC.AA. (~=Canarias)

% Para cada K-cluster se geo-localizan en el mapa todas sus ciudades
for K=1:minK
    geoshow(TfilterCA(TfilterCA.Group_ID==K,:).Latitud,...
            TfilterCA(TfilterCA.Group_ID==K,:).Longitud,...
            'DisplayType','Point',...
            'MarkerFaceColor',palette(K,:),...
            'MarkerEdgeColor','Black','LineWidth',0.1,...
            'Marker', 'o');        
end

% Leyenda din�mica en funci�n del minK obtenido
for K=1:minK
    p(K)= patch(NaN, NaN, palette(K,:));
    p(K).DisplayName= strcat('K=', int2str(K));
end
legend(p);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M a p a   a u t o n � m i c o   d e   I s l a s   C a n a r i a s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(6)
palette= colormap(hsv(minK));     % Se reinicia la paleta de color
ax= worldmap('Canary Islands');     % Selecci�n del area geogr�fica
setm(ax, 'FFaceColor', oceanClr,... % Parametrizaci�n del mapa
    'MapProjection','mercator')     % Proyecci�n plana

% Se muestra el �rea geogr�fica y el mapa auton�mico
geoshow(ax, land, 'FaceColor', landClr)
geoshow(ccaa)
mlabel; plabel; gridm;              % Sin elementos auxiliares en el mapa
TfilterCA= T(T.Codigo_CA==5,:);     % Filtramos s�lo Islas Canarias

% Para cada K-cluster se geo-localizan en el mapa todas sus ciudades
for K=1:minK
    geoshow(TfilterCA(TfilterCA.Group_ID==K,:).Latitud,...
            TfilterCA(TfilterCA.Group_ID==K,:).Longitud,...
            'DisplayType','Point',...
            'MarkerFaceColor',palette(K,:),...
            'MarkerEdgeColor','Black','LineWidth',0.1,...
            'Marker', 'o');
end

% Leyenda din�mica en funci�n del minK obtenido
for K=1:minK
    p(K)= patch(NaN, NaN, palette(K,:));
    p(K).DisplayName= strcat('K=', int2str(K));
end
legend(p);
