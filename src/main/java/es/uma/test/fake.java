package es.uma.test;

public class fake {
    public fake(){}

    public String getMessage(){
        return "{\n" +
                "  \"id\": \"5f217400922a2c0183f6c47b\",\n" +
                "  \"timestamp\": \"1596100510\",\n" +
                "  \"name\": \"PredictionRequestedEvent\",\n" +
                "  \"version\": \"1\",\n" +
                "  \"executionId\": \"5f21671a401942005039dbb0\",\n" +
                "  \"operation\": \"forecast\",\n" +
                "  \"model\": {\n" +
                "    \"type\": \"modelA\",\n" +
                "    \"from\": \"2020-07-29T12:10:02.275Z\",\n" +
                "    \"to\": \"2020-11-2T12:10:02.275Z\",\n" +
                "    \"horizon\": \"2d\",\n" +
                "    \"refKpi\": \"fiwoo:kpi:5f216a33ca66b3007dc4d828\"\n" +
                "  },\n" +
                "  \"origins\": [\n" +
                "    {\n" +
                "      \"type\": \"etl\",\n" +
                "      \"id\": {\n" +
                "        \"urn\": \"fiwoo:etl:5f21671a401942005039dbb0\"\n" +
                "      },\n" +
                "      \"datasources\": [\n" +
                "        {\n" +
                "          \"kind\": \"proccesed\",\n" +
                "          \"source\": {\n" +
                "            \"type\": \"mongodb\",\n" +
                "            \"host\": \"127.0.0.1\",\n" +
                "            \"port\": 27017,\n" +
                "            \"name\": \"um\",\n" +
                "            \"collection\": \"device\"\n" +
                "          },\n" +
                "          \"data\": {\n" +
                "            \"mime\": \"database/mongodb+collection\",\n" +
                "            \"schema\": {\n" +
                "              \"type\": \"json-ngsiv2\",\n" +
                "              \"definition\": {\n" +
                "                \"device_id\": {\n" +
                "                  \"type\": \"string\"\n" +
                "                },\n" +
                "                \"field_a\": {\n" +
                "                  \"type\": \"integer\"\n" +
                "                },\n" +
                "                \"field_b\": {\n" +
                "                  \"type\": \"string\"\n" +
                "                },\n" +
                "                \"timestamp\": {\n" +
                "                  \"type\": \"date\"\n" +
                "                }\n" +
                "              }\n" +
                "            }\n" +
                "          }\n" +
                "        }\n" +
                "      ]\n" +
                "    }\n" +
                "  ],\n" +
                "  \"output\": {\n" +
                "    \"source\": {\n" +
                "      \"type\": \"mongodb\",\n" +
                "      \"host\": \"localhost\",\n" +
                "      \"port\": 27017,\n" +
                "      \"name\": \"um\",\n" +
                "      \"collection\": \"device2\"\n" +
                "    },\n" +
                "    \"data\": {\n" +
                "      \"mime\": \"database/mongodb+collection\",\n" +
                "      \"schema\": {\n" +
                "        \"type\": \"json-ngsiv2\",\n" +
                "        \"definition\": {\n" +
                "          \"date\": {\n" +
                "            \"type\": \"date\"\n" +
                "          },\n" +
                "          \"predicted_field_a\": {\n" +
                "            \"type\": \"integer\"\n" +
                "          },\n" +
                "          \"predicted_field_b\": {\n" +
                "            \"type\": \"string\"\n" +
                "          },\n" +
                "          \"sub-model\": {\n" +
                "            \"type\": \"string\"\n" +
                "          }\n" +
                "        }\n" +
                "      }\n" +
                "    }\n" +
                "  }\n" +
                "}";
    }

    public void sendMessage(String s){}
}
