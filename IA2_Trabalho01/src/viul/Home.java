package viul;

import ia2_trab1.Investidor;

/**
 *
 * @author Bruno
 */
public class Home extends javax.swing.JFrame {

    Investidor i;
    ListaEmpresas le;
    ResumoIndividual ri;
    
    public Home(Investidor i) {
        initComponents();
        
        this.i = i;
        le = new ListaEmpresas(i);
        ri = new ResumoIndividual(i);
        this.panelTabs.setComponentAt(0, le);
        this.panelTabs.setComponentAt(1, ri);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jSeparator1 = new javax.swing.JSeparator();
        jLabel1 = new javax.swing.JLabel();
        panelTabs = new javax.swing.JTabbedPane();
        pnlLista = new javax.swing.JPanel();
        pnlResumoIndividual = new javax.swing.JPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jLabel1.setText("Investidor");

        javax.swing.GroupLayout pnlListaLayout = new javax.swing.GroupLayout(pnlLista);
        pnlLista.setLayout(pnlListaLayout);
        pnlListaLayout.setHorizontalGroup(
            pnlListaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 775, Short.MAX_VALUE)
        );
        pnlListaLayout.setVerticalGroup(
            pnlListaLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 521, Short.MAX_VALUE)
        );

        panelTabs.addTab("Empresas de Interesse", pnlLista);

        javax.swing.GroupLayout pnlResumoIndividualLayout = new javax.swing.GroupLayout(pnlResumoIndividual);
        pnlResumoIndividual.setLayout(pnlResumoIndividualLayout);
        pnlResumoIndividualLayout.setHorizontalGroup(
            pnlResumoIndividualLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 775, Short.MAX_VALUE)
        );
        pnlResumoIndividualLayout.setVerticalGroup(
            pnlResumoIndividualLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 521, Short.MAX_VALUE)
        );

        panelTabs.addTab("Resumo Individual", pnlResumoIndividual);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(jSeparator1)
                    .addComponent(jLabel1, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(panelTabs, javax.swing.GroupLayout.Alignment.LEADING))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jSeparator1, javax.swing.GroupLayout.PREFERRED_SIZE, 10, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(panelTabs)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel jLabel1;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JTabbedPane panelTabs;
    private javax.swing.JPanel pnlLista;
    private javax.swing.JPanel pnlResumoIndividual;
    // End of variables declaration//GEN-END:variables
}
