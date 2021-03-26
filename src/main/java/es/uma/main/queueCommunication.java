package es.uma.main;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.GetResponse;
import es.uma.models.IncomingMessage;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeoutException;

public class queueCommunication {
    private static final String RCPT_EXCHANGE_NAME = "ex.runtime-management";
    private static final String SEND_EXCHANGE_NAME = "ex.predictions";
    private static final String ROUTING_KEY = "actions.forecast";
    private static final String BINDINGS = "ms.predictions";
    private static final String TOPIC = RCPT_EXCHANGE_NAME + "/" + BINDINGS;

    private ConnectionFactory factory;
    private Connection connection;
    private Channel send_channel;
    private Channel recv_channel;

    public queueCommunication(String user, String passwd, String host, String vhost, Integer port) {
        setFactory(new ConnectionFactory());
        getFactory().setHost(host);
        getFactory().setUsername(user);
        getFactory().setPassword(passwd);
        getFactory().setVirtualHost(vhost);
        getFactory().setPort(port);

        try {
            setConnection(getFactory().newConnection());
            setRecv_channel(getConnection().createChannel());
            setSend_channel(getConnection().createChannel());

            getSend_channel().exchangeDeclare(SEND_EXCHANGE_NAME, "topic");
            getRecv_channel().queueDeclare(TOPIC, true, false, false, null);
        } catch (TimeoutException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void sendMessage(String message, long deliveryTag){
        sendMessage(message, deliveryTag, ROUTING_KEY);
    }

    public void sendMessage(String message, long deliveryTag, String routingKey){
        try {
            getSend_channel().basicPublish(SEND_EXCHANGE_NAME, routingKey, null, message.getBytes("UTF-8"));
            getRecv_channel().basicAck(deliveryTag, false);
        } catch (IOException e) {
            System.err.println("Error sending a message to queue");
            e.printStackTrace();
        }
        System.out.println(" [x] Sent '" + routingKey + "':'" + message + "'");
    }

    public IncomingMessage getMessage(){
        while(true){
            GetResponse response = null;
            try {
                response = getRecv_channel().basicGet(TOPIC, false);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (response != null){
                IncomingMessage m = new IncomingMessage();
                m.deliveryTag = response.getEnvelope().getDeliveryTag();
                m.message = new String(response.getBody(), StandardCharsets.UTF_8);
                return m;
            }
        }
    }

    public Channel getSend_channel(){
        if(!getConnection().isOpen()) {
            try {
                setConnection(getFactory().newConnection());
                setSend_channel(getConnection().createChannel());
                send_channel.exchangeDeclare(SEND_EXCHANGE_NAME, "topic");
            } catch (IOException e) {
                e.printStackTrace();
            } catch (TimeoutException e) {
                e.printStackTrace();
            }
        }
        return send_channel;
    }

    public void setSend_channel(Channel send_channel){
        this.send_channel = send_channel;
    }

    public Channel getRecv_channel(){
        if(!getConnection().isOpen()) {
            try {
                setConnection(getFactory().newConnection());
                setRecv_channel(getConnection().createChannel());
                recv_channel.queueDeclare(TOPIC, true, false, false, null);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (TimeoutException e) {
                e.printStackTrace();
            }
        }
        return recv_channel;
    }

    public void setRecv_channel(Channel recv_channel){
        this.recv_channel = recv_channel;
    }

    public Connection getConnection(){
        return connection;
    }

    public void setConnection(Connection connection){
        this.connection = connection;
    }

    public ConnectionFactory getFactory(){
        return factory;
    }

    public void setFactory(ConnectionFactory factory) {
        this.factory = factory;
    }
}
